import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.nn import Parameter
from torch.nn import MSELoss, L1Loss, SmoothL1Loss, CrossEntropyLoss
from torch.distributions.binomial import Binomial


import pandas as pd
import numpy as np
import math
from scipy.stats import spearmanr
from sklearn.metrics import precision_score, f1_score, recall_score

from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm_n

from collections import Iterable, defaultdict
import itertools

#from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.commands.elmo import ElmoEmbedder

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

#elmo = Elmo(options_file, weight_file, 1, dropout=0, requires_grad=False)  #using 1 layer of representation
elmo = ElmoEmbedder(options_file, weight_file, cuda_device=2)
# elmo = ElmoEmbedder(options_file, weight_file)

class TemporalModel(torch.nn.Module):
    '''
     A class to run attention models on tuned ELMO word embeddings stacked up with MLP layers
     
     # Each of event_attention, dur_attention, rel_attention can take three values:
        - root
        - constant
        - param
    '''
    def __init__(self, embedding_size=1024, 
                tune_embed_size=300,
                mlp_dropout = 0.5,
                mlp_activation="tanh",
                duration_distr = False,
                attention=True, event_attention='root', dur_attention = 'param', 
                rel_attention = 'param', dur_MLP_sizes = [24], fine_MLP_sizes = [24],
                coarse_MLP_sizes = [24], dur_output_size = 11, fine_output_size = 4,
                coarse_output_size = 7,
                device=torch.device(type="cpu") ):
        super().__init__()

        self.device = device
        # self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.tuned_embed_size = tune_embed_size
        self.event_attention = event_attention
        self.dur_attention = dur_attention
        self.rel_attention = rel_attention
        self.mlp_activation = mlp_activation
        self.duration_distr = duration_distr
        self.dur_output_size = dur_output_size

        #initialize embedding-tuning MLP
        self.tuned_embed_MLP = nn.Linear(self.embedding_size*3, self.tuned_embed_size)
        self.mlp_dropout =  nn.Dropout(mlp_dropout) 
        #Initialize attention parameters
        self._init_attention()

        # initialize MLP layers
        self.linear_maps = nn.ModuleDict()

        if self.duration_distr:
            self._init_MLP(tune_embed_size,dur_MLP_sizes,1, param="duration")
        else:
            self._init_MLP(tune_embed_size,dur_MLP_sizes,dur_output_size, param="duration")

        self._init_MLP(tune_embed_size, fine_MLP_sizes, fine_output_size, param="fine")
        self._init_MLP(tune_embed_size, coarse_MLP_sizes,coarse_output_size, param="coarse")

    def _init_attention(self):

        #### Event attention ####
        if self.event_attention == "root":
            pass

        elif self.event_attention == "constant":
            self.event_att_map = torch.nn.Linear(self.tuned_embed_size, 1, bias=False)

        elif self.event_attention == "param":
            self.event_att_map = torch.nn.Linear(self.tuned_embed_size, self.tuned_embed_size)

        #### Duration attention ####
        if self.dur_attention == "root":
            pass

        elif self.dur_attention == "constant":
            self.dur_att_map = torch.nn.Linear(self.tuned_embed_size, 1, bias=False)

        elif self.dur_attention == "param" and self.event_attention == "root":
            self.dur_att_map = torch.nn.Linear(self.tuned_embed_size, self.tuned_embed_size)

        elif self.dur_attention == "param" and self.event_attention != "root":
            self.dur_att_map = torch.nn.Linear(self.tuned_embed_size*2, self.tuned_embed_size)

        #### Relation attention ####
        if self.rel_attention == "root":
            pass

        elif self.rel_attention == "constant":
            self.rel_att_map =  torch.nn.Linear(self.tuned_embed_size, 1, bias=False)

        elif self.rel_attention == "param" and self.event_attention == "root":
            self.rel_att_map = torch.nn.Linear(self.tuned_embed_size*2, self.tuned_embed_size)

        elif self.rel_attention == "param" and self.event_attention != "root":
            self.rel_att_map = torch.nn.Linear(self.tuned_embed_size*4, self.tuned_embed_size)


    def _init_MLP(self, input_size, hidden_sizes, output_size, param=None):
        '''
        Initialise MLP or regression parameters
        '''
        self.linear_maps[param] = nn.ModuleList()

        if param=="duration":
            if self.event_attention == "root" and self.dur_attention=="root":
                input_size = input_size
            elif self.event_attention == "root" and self.dur_attention !="root":
                input_size = input_size*2
            elif self.event_attention != "root" and self.dur_attention=="root":
                input_size = input_size*2
            elif self.event_attention != "root" and self.dur_attention !="root":
                input_size = input_size*3

        else:
            if self.event_attention == "root" and self.rel_attention == "root":
                input_size = input_size*2
            elif self.event_attention == "root" and self.rel_attention != "root":
                input_size = input_size*3
            elif self.event_attention != "root" and self.rel_attention == "root":
                input_size = input_size*4
            elif self.event_attention != "root" and self.rel_attention != "root":
                input_size = input_size*5

        for h in hidden_sizes:
            linmap = torch.nn.Linear(input_size, h)
            linmap = linmap.to(self.device)
            self.linear_maps[param].append(linmap)
            input_size = h

        linmap = torch.nn.Linear(input_size, output_size)
        linmap = linmap.to(self.device)
        self.linear_maps[param].append(linmap)

    def forward(self, structures, spans_idxs, root_idxs):
        '''
        Input: 1. structures: A list of list of words
               2. idxs: A list of list of span indexes 
        
        Inputs are run through multiple attention layers followed by MLP layers
        '''
        inputs = elmo.batch_to_embeddings(structures)[0].to(self.device)
        #Concatenate ELMO's layers
        inputs = inputs.view(inputs.size()[0], inputs.size()[2], -1).to(self.device)

        #tune embeddings into lower dim:
        inputs = self._tune_embeddings(inputs)

        #pre-process inputs
        inputs = self._preprocess_inputs(inputs)

        ## Extract pred1, pred2 indexes:
        pred1_r_idxs = [x for x,y in root_idxs]
        pred2_r_idxs = [y for x,y in root_idxs]
        pred1_spans = [x for x,y in spans_idxs]
        pred2_spans = [y for x,y in spans_idxs]

        #Run Event attention on inputs based on attention type:
        pred1_out = self._run_event_attention(inputs, pred1_spans, pred1_r_idxs)
        pred2_out = self._run_event_attention(inputs, pred2_spans, pred2_r_idxs)
            
        #Run duration attention on outputs from event attention
        pred1_dur = self._run_duration_attention(inputs, pred1_out)
        pred2_dur = self._run_duration_attention(inputs, pred2_out)

        #Run through relative_temporal type:
        rel_output = self._run_relation_attention(inputs, pred1_out, pred2_out)

        #Run Duration-MLP
        pred1_dur = self._run_regression(pred1_dur, param="duration", activation=self.mlp_activation)
        pred2_dur = self._run_regression(pred2_dur, param="duration", activation=self.mlp_activation)

        if self.duration_distr:
            pred1_dur = self._binomial_dist(pred1_dur)
            pred2_dur = self._binomial_dist(pred2_dur)

        #Run Fine-grained-MLP
        fine_output = self._run_regression(rel_output, param="fine", activation=self.mlp_activation)
        # fine_output[:,1] = fine_output[:,0] + fine_output[:,1]
        # fine_output[:,3] = fine_output[:,2] + fine_output[:,3]
        # fine_output = torch.sigmoid(fine_output)

        #Run Coarse-grained MLP
        coarse_output = self._run_regression(rel_output, param="coarse", activation=self.mlp_activation)

        y_hat = (pred1_dur, pred2_dur, fine_output, coarse_output)

        return y_hat

    def _binomial_dist(self, pred_dur):
        '''
        Input: A tensor with dimension: batch_size x 1
        Output: A tensor with dimension: batch_size x 11 
        Binomial Prob distribution for a given duration value 

        '''
        pred_dur = torch.sigmoid(pred_dur)

        dur_size = self.dur_output_size
        batch_size = pred_dur.size()[0]

        ans = torch.zeros((batch_size, dur_size)).to(self.device)
        durations = torch.tensor(range(dur_size), dtype=torch.float).to(self.device)

        for i,m in enumerate(pred_dur):
            bin_class = Binomial(total_count=10, probs=m)
            ans[i,:] = bin_class.log_prob(durations)

        return ans

    def _tune_embeddings(self, inputs):
        return torch.tanh(self.tuned_embed_MLP(inputs))

    def _extract_root_inputs(self, inputs, root_idxs):
        '''
        Inputs:
        1. inputs: embeddings for full sentences in a batch
                        Shape: batch_size x max_batch_len x embedding size
        2. root_idxs: indexes of the predicate's root in each sentence

         Output:
        1. Embeddings of predicate's root
            Shape: batch_size x embedding_size
        '''
        batch_size  = inputs.size()[0]
        root_inputs = torch.zeros((batch_size, self.tuned_embed_size), 
                                    dtype=torch.float, device=self.device)

        for sent_idx in range(batch_size):
            root_idx = root_idxs[sent_idx]
            root_inputs[sent_idx] = inputs[sent_idx][root_idx]

        return root_inputs

    def _extract_span_inputs(self, inputs, span_idxs):
        '''
        Extract embeddings for a span in the sentence
        
        For a mini-batch, keeps the length of span equal to the length 
        max span in that batch
        '''
        batch_size = inputs.size()[0]
        span_lengths = [len(x) for x in span_idxs]
        max_span_len = max(span_lengths)
        
        span_embeds = torch.zeros((batch_size, max_span_len, self.tuned_embed_size), 
                                  dtype=torch.float, device=self.device)
        
        for sent_idx in range(batch_size):
            m=0
            for span_idx in span_idxs[sent_idx]:
                span_embeds[sent_idx][m] = inputs[sent_idx][span_idx]
                m+=1
                
        return span_embeds

    def _run_event_attention(self, inputs, pred1_spans, pred1_r_idxs):
        '''
        Input: An input tensor with dimension:
             (batch_size x max_sentence_len x embedding_size)

        Output: pred1 output emmbeddings after running 
                the corresponding attention types
                
                Shape(pred1_out): (batch_size x embedding_size)
                                        OR
                                  (batch_size x 2*embedding_size)

        '''
        pred1_root = self._extract_root_inputs(inputs, pred1_r_idxs)

        batch_size = inputs.size()[0]

        if self.event_attention=="root":
            return pred1_root

        elif self.event_attention == "constant" :
            pred1_span_inputs = self._extract_span_inputs(inputs, pred1_spans)
            att_raw = self.event_att_map(pred1_span_inputs)
            att_raw = att_raw.masked_fill(att_raw[:, :, 0:1] == 0, -float('inf'))
            att = F.softmax(att_raw.view(batch_size, pred1_span_inputs.shape[1]), dim=1)
            pred1_context = torch.bmm(att[:, None, :], pred1_span_inputs).squeeze()
           
            return torch.cat((pred1_root, pred1_context), dim=1)

        elif self.event_attention == "param":
            pred1_span_inputs = self._extract_span_inputs(inputs, pred1_spans)
            att_span = self.event_att_map(pred1_root)
            att_span = self._regression_nonlinearity(att_span)
            att_raw = torch.bmm(pred1_span_inputs, att_span[:, :, None])
            att_raw = att_raw.masked_fill(att_raw[:, :, 0:1] == 0, -float('inf'))
            att = F.softmax(att_raw.view(batch_size, pred1_span_inputs.shape[1]), dim=1)
            pred1_context = torch.bmm(att[:, None, :], pred1_span_inputs).squeeze()
           
            return torch.cat((pred1_root, pred1_context), dim=1)

    def _run_duration_attention(self, inputs, pred_in):
        '''
        Input:
        1. inputs Embeddings of the whole sentence
        2. pred_in: embeddings of pred_i output from event_attention on pred_i
        
        Output:

        '''
        batch_size = inputs.size()[0]
        if self.dur_attention == "root":
            return pred_in

        elif self.dur_attention == "constant":
            att_raw = self.dur_att_map(inputs)
            att_raw = att_raw.masked_fill(att_raw[:, :, 0:1] == 0, -float('inf'))
            att = F.softmax(att_raw.view(batch_size, inputs.shape[1]), dim=1)
            dur_context = torch.bmm(att[:, None, :], inputs).squeeze()

            return torch.cat((pred_in, dur_context), dim=1)
            
        elif self.dur_attention == "param":
            att_span = self.dur_att_map(pred_in)
            att_span = self._regression_nonlinearity(att_span)
            # print("Attention span dim: {}".format(att_span.size()))
            att_raw = torch.bmm(inputs, att_span[:, :, None])
            # print("Attention raw dim: {}".format(att_raw.size()))
            # print("Attention raw softm dim: {}".format(att_raw.view(batch_size, inputs.shape[1]).size()))
            att_raw = att_raw.masked_fill(att_raw[:, :, 0:1] == 0, -float('inf'))
            att = F.softmax(att_raw.view(batch_size, inputs.shape[1]), dim=1)
            dur_context = torch.bmm(att[:, None, :], inputs).squeeze()

            return torch.cat((pred_in, dur_context), dim=1)


    def _run_relation_attention(self, inputs, pred1_in, pred2_in):
        '''
        Inputs:
        1. inputs: Embeddings of the whole sentence
        2. pred1_out: embeddings of pred1 output from event_attention on pred1
        3. pred2_out: embeddings of pred2 output from event_attention on pred2

        Output:
        Final attention-layer output combining both the predicates 
        as per the relation-attention
        '''
        batch_size = inputs.size()[0]
        if self.rel_attention == "root":
            return torch.cat((pred1_in, pred2_in), dim=1)

        elif self.rel_attention == "constant":
            att_raw = self.rel_att_map(inputs)
            att_raw = att_raw.masked_fill(att_raw[:, :, 0:1] == 0, -float('inf'))
            att = F.softmax(att_raw.view(batch_size, inputs.shape[1]), dim=1)
            rel_context = torch.bmm(att[:, None, :], inputs).squeeze()

            return torch.cat((pred1_in, pred2_in, rel_context), dim=1)

        elif self.rel_attention == "param":
            att_span = self.rel_att_map(torch.cat((pred1_in, pred2_in), dim=1))
            att_span = self._regression_nonlinearity(att_span)
            att_raw = torch.bmm(inputs, att_span[:, :, None])
            att_raw = att_raw.masked_fill(att_raw[:, :, 0:1] == 0, -float('inf'))
            att = F.softmax(att_raw.view(batch_size, inputs.shape[1]), dim=1)
            rel_context = torch.bmm(att[:, None, :], inputs).squeeze()

            return torch.cat((pred1_in, pred2_in, rel_context), dim=1)

    def _preprocess_inputs(self, inputs):
        """Apply some function(s) to the input embeddings
        This is included to allow for an easy preprocessing hook for
        RNNRegression subclasses. For instance, we might want to
        apply a tanh to the inputs to make them look more like features
        """
        return inputs
 
    def _run_regression(self, h_last, param=None, activation=None):

        for i, linear_map in enumerate(self.linear_maps[param]):
            if i:
                if activation == "sigmoid":
                    h_last = torch.sigmoid(h_last)
                    h_last = self.mlp_dropout(h_last)
                elif activation == "relu":
                    h_last = F.relu(h_last)
                    h_last = self.mlp_dropout(h_last)                  
                else:
                    h_last = torch.tanh(h_last)
                    h_last = self.mlp_dropout(h_last)

            h_last = linear_map(h_last)

        if param=="fine":
            return torch.sigmoid(h_last)
        else:
            return h_last
        
    
    def _postprocess_outputs(self, outputs):
        """Apply some function(s) to the output value(s)"""
        return outputs.squeeze()

    def _regression_nonlinearity(self, x):
        return torch.tanh(x)


class TemporalTrainer(object):
    
    loss_function_map = {"linear": MSELoss,
                         "robust": L1Loss,
                         "robust_smooth": SmoothL1Loss,
                         "multinomial": CrossEntropyLoss}
    
    def __init__(self, regression_type="robust",
                 optimizer_class=torch.optim.Adam,
                 device=torch.device(type="cpu"), 
                 model_file_prefix = "",
                 train_batch_size = 4,
                 predict_loss_batch_size = 256,
                 predict_batch_size = 256,
                 epochs=5,
                 logmax = 0.9999999,
                 model_file_suffix = ".pth",
                **kwargs):
        
        self.epochs = epochs
        #self.rnn_class = rnn_class
        self.device = device
        self.train_batch_size = train_batch_size
        self.predict_batch_size = predict_batch_size
        self.predict_loss_batch_size = predict_loss_batch_size

        self.best_model_file = model_file_prefix + "model_" + kwargs['event_attention'] +  \
                                "_" + kwargs['dur_attention'] +  \
                                "_" + kwargs['rel_attention'] + \
                                "_" + "_".join([str(x) for x in kwargs['dur_MLP_sizes']]) + \
                                "_" + str(kwargs['mlp_dropout']) + "_" + kwargs['mlp_activation'] + \
                                "_binom_" + str(kwargs['duration_distr']) + "_coarsesize_" + \
                                str(kwargs['coarse_output_size']) +\
                                model_file_suffix

        self._regression_type = regression_type
        self._optimizer_class = optimizer_class
        self._init_kwargs = kwargs
        self._continuous = regression_type != "multinomial"
        self.logmax = logmax


    def _initialize_trainer_model(self):
        self._model = TemporalModel(device=self.device,
                                             **self._init_kwargs)
        
        self._model = self._model.to(self.device)
        self.fine_loss = L1Loss(reduction='none').to(self.device)


    def _compute_class_weights(self, variable):
        '''
        Computes class weights for a categorical variable
        and outputs a tensor of class-weights

        Required incase of unbalanced classes
        '''
        class_dict = defaultdict(int)
        for item in variable:
            class_dict[item]+=1
        class_weights = sorted([(cl,1/num) for cl, num in class_dict.items()], key=lambda x: x[0])
        class_weights = [y for x,y in class_weights]
        class_weights = torch.FloatTensor(class_weights).to(self.device)
        class_weights = class_weights / class_weights.sum(0).expand_as(class_weights)

        return class_weights

    def _lsts_to_tensors(self, *args, param=None):
        '''
        Input: list1, list2,......

        Output: [Tensor(list1), tensor(list2),....]

        '''
        if param=="float":
            return [torch.from_numpy(np.array(arg)).float().to(self.device) for arg in args]
        else:
            return [torch.from_numpy(np.array(arg, dtype="int64")).to(self.device) for arg in args]


    def _custom_temporal_loss(self, model_out, durations, sliders, time_ml, 
                                pred1_conf,pred2_conf, rel_conf):
        '''
        Calculate L1 to L6 as described in the paper
        '''
        batch_size = len(durations)
        out_p1_d, out_p2_d, out_f, out_c = model_out

        #Store actual_y into tensors
        pred1_durs = [p for p,q in durations]
        pred2_durs = [q for p,q in durations]

        pred1_durs, pred2_durs, time_ml = self._lsts_to_tensors(pred1_durs,pred2_durs,
                                                                 time_ml)

        pred1_conf, pred2_conf, rel_conf = self._lsts_to_tensors(pred1_conf,pred2_conf,
                                                                 rel_conf, param="float")

        b_p1 = [p1[0] for p1, p2 in sliders]
        e_p1 = [p1[1] for p1, p2 in sliders]
        b_p2 = [p2[0] for p1, p2 in sliders]
        e_p2 = [p2[1] for p1, p2 in sliders]

        b_p1, e_p1, b_p2, e_p2 = self._lsts_to_tensors(b_p1, e_p1, b_p2, e_p2, 
                                                        param="float")

        ## Duration and Coarse-grained Losses
        L1_p1 = self.duration_loss(out_p1_d, pred1_durs)
        L1_p1 = torch.mm(pred1_conf.unsqueeze(0), L1_p1.unsqueeze(1))/batch_size

        L1_p2 = self.duration_loss(out_p2_d, pred2_durs)
        L1_p2 = torch.mm(pred2_conf.unsqueeze(0), L1_p2.unsqueeze(1))/batch_size
        # print("L1_p1 {},  L1_p2: {}".format(L1_p1, L1_p2))

        L6 = self.coarse_loss(out_c, time_ml)
        L6 = torch.mm(rel_conf.unsqueeze(0), L6.unsqueeze(1))/batch_size

        # print("L6 : {}".format(L6))
        max_tensor = torch.ones((batch_size,)).to(self.device)
        max_tensor = max_tensor.new_full((batch_size,1), self.logmax)
        max_tensor = max_tensor.view(batch_size)

        ## Fine-grained Losses
        L2 = torch.min(self.fine_loss(out_f[:, 0]-out_f[:, 2], b_p1-b_p2), max_tensor)
        L3 = torch.min(self.fine_loss(out_f[:, 0]+out_f[:, 1]-out_f[:, 2], e_p1-b_p2), max_tensor)
        L4 = torch.min(self.fine_loss(out_f[:, 2]+out_f[:, 3]-out_f[:, 0], e_p2-b_p1), max_tensor)
        L5 = torch.min(self.fine_loss(out_f[:, 0]+out_f[:, 1]-out_f[:, 2]-out_f[:, 3], e_p1-e_p2), max_tensor)
        # L2 = torch.min(self.fine_loss(out_f[:, 0]-out_f[:, 2], b_p1-b_p2), max_tensor)
        # L3 = torch.min(self.fine_loss(out_f[:, 1]-out_f[:, 2], e_p1-b_p2), max_tensor)
        # L4 = torch.min(self.fine_loss(out_f[:, 3]-out_f[:, 0], e_p2-b_p1), max_tensor)
        # L5 = torch.min(self.fine_loss(out_f[:, 1]-out_f[:, 3], e_p1-e_p2), max_tensor)

        # print("L2 {}, L3 {}, L4 {}, L5 {}".format(L2, L3, L4, L5))

        L2to5 = sum([-torch.log(1-L2), -torch.log(1-L3), 
            -torch.log(1-L4), -torch.log(1-L5)])/4

        L2to5 = torch.mm(rel_conf.unsqueeze(0), L2to5.unsqueeze(1))/batch_size
        # print("L2to5 {}".format(L2to5))
        # print("final loss: {}".format(sum([(L1_p1+L1_p2)/2,  L6 , L2to5])/3))

        return (sum([(L1_p1+L1_p2)/2,  L6, L2to5])/3).squeeze()

    def fit(self, train_X, train_Y, dev, verbosity=1, **kwargs):
        """Fit the LSTM regression
        Parameters
        ----------
        X : iterable(iterable(object))
            a matrix of structures (independent variables) with rows
            corresponding to a particular kind of RNN
        Y : numpy.array(Number)
            a matrix of dependent variables
        batch_size : int (default: 100)
        verbosity : int (default: 1)
            how often to print metrics (never if 0)
        """
        self._X,  self._Y = train_X, train_Y

        durations = [x for x,y,z,p,q,r in self._Y]
        sliders = [y for x,y,z,p,q,r in self._Y]
        time_ml = [z for x,y,z,p,q,r in self._Y]
        pred1_conf = [p for x,y,z,p,q,r in self._Y]
        pred2_conf = [q for x,y,z,p,q,r in self._Y]
        rel_conf = [r for x,y,z,p,q,r in self._Y]

        dev_x, dev_y = dev
        
        self._initialize_trainer_model()  
        
        #Loss for duration:
        raw_durations = itertools.chain(*durations)
        # dur_wts = self._compute_class_weights(raw_durations)
        self.duration_loss = CrossEntropyLoss(reduction='none')
        # print("Duration weights: {}".format(dur_wts))
        self.duration_loss = self.duration_loss.to(self.device)

        #Loss for time_ml class:
        # coarse_wts = self._compute_class_weights(time_ml)
        self.coarse_loss = CrossEntropyLoss(reduction='none')
        # print("Coarse weights: {}".format(coarse_wts))
        self.coarse_loss = self.coarse_loss.to(self.device)


        print("########## .   Model Parameters   ##############")
        for name,param in self._model.named_parameters():     
            if param.requires_grad:
                print(name, param.shape) 
                print("\n")
        print("##############################################") 

        parameters = [p for p in self._model.parameters() if p.requires_grad]
        optimizer = self._optimizer_class(parameters, **kwargs)
        
        total_obs = len(self._X)
        dev_obs = len(dev_x)
        
        dev_accs = []
        train_accs = []
        best_val_acc = -float('inf')
        best_val_loss = float('inf')
        bad_count = 0

        for epoch in range(self.epochs):
            # Turn on training mode which enables dropout.
            self._model.train()
            
            bidx_i = 0
            bidx_j =self.train_batch_size
            
            tqdm.write("Running Epoch: {}".format(epoch+1))
            
            #time print
            pbar = tqdm(total = total_obs//self.train_batch_size)
            
            while bidx_j < total_obs:
                words = [p for p,q,r in self._X[bidx_i:bidx_j]]
                spans = [q for p,q,r in self._X[bidx_i:bidx_j]]
                roots = [r for p,q,r in self._X[bidx_i:bidx_j]]
                
                #Zero grad
                optimizer.zero_grad()

                #Calculate Loss
                model_out  = self._model(words, spans, roots)   

                curr_loss = self._custom_temporal_loss(model_out, 
                                            durations[bidx_i:bidx_j],
                                            sliders[bidx_i:bidx_j],
                                            time_ml[bidx_i:bidx_j],
                                            pred1_conf[bidx_i:bidx_j],
                                            pred2_conf[bidx_i:bidx_j],
                                            rel_conf[bidx_i:bidx_j])
                #Backpropagate
                curr_loss.backward()
                optimizer.step()
                bidx_i = bidx_j
                bidx_j = bidx_i + self.train_batch_size
                
                if bidx_j >= total_obs:
                    words = [p for p,q,r in self._X[bidx_i:bidx_j]]
                    spans = [q for p,q,r in self._X[bidx_i:bidx_j]]
                    roots = [r for p,q,r in self._X[bidx_i:bidx_j]]

                    #Zero grad
                    optimizer.zero_grad()
                    
                    #Calculate Loss
                    model_out  = self._model(words, spans, roots)   

                    curr_loss = self._custom_temporal_loss(model_out, 
                                            durations[bidx_i:bidx_j],
                                            sliders[bidx_i:bidx_j],
                                            time_ml[bidx_i:bidx_j],
                                            pred1_conf[bidx_i:bidx_j],
                                            pred2_conf[bidx_i:bidx_j],
                                            rel_conf[bidx_i:bidx_j])
                
                    #Backpropagate
                    curr_loss.backward()
                    optimizer.step()
                    
                pbar.update(1)
                    
            pbar.close()
            
            
            ## Dev_loss:
            #dev_predicts = self.predict(dev_x)
            dev_loss = self._predict_loss(dev_x, dev_y)
        
            # Save the model if the validation loss is the best we've seen so far.

            if dev_loss < best_val_loss:
                with open(self.best_model_file, 'wb') as f:
                    torch.save(self._model.state_dict(), f)
                best_val_loss = dev_loss
            
            if epoch:
                if dev_loss > dev_accs[-1]:
                    bad_count+=1
                else:
                    bad_count=0
            
            tqdm.write("Epoch No.: {} Last Train-minibatch Loss: {}".format(epoch+1, curr_loss))
            tqdm.write("Total Dev Loss: {}".format(dev_loss))
            tqdm.write("\n")
            dev_accs.append(dev_loss.detach())

            if bad_count >=2:
                break
            
            
        return dev_accs

    def _predict_loss(self, data_x, data_y):
        """Predict the temporal loss using the model
        Parameters
        ----------
        data_x
        data_y
        """
        # Turn on evaluation mode which disables dropout.
        self._model.eval()
        
        durations = [x for x,y,z,p,q,r in data_y]
        sliders = [y for x,y,z,p,q,r in data_y]
        time_ml = [z for x,y,z,p,q,r in data_y]
        pred1_conf = [p for x,y,z,p,q,r in data_y]
        pred2_conf = [q for x,y,z,p,q,r in data_y]
        rel_conf = [r for x,y,z,p,q,r in data_y]

        total_loss = 0

        with torch.no_grad():  
            bidx_i = 0
            bidx_j = self.predict_loss_batch_size
            total_obs = len(data_x)
            
            while bidx_j < total_obs:
                words = [p for p,q,r in data_x[bidx_i:bidx_j]]
                spans = [q for p,q,r in data_x[bidx_i:bidx_j]]
                roots = [r for p,q,r in data_x[bidx_i:bidx_j]]
                model_out  = self._model(words, spans, roots)   
                curr_loss = self._custom_temporal_loss(model_out, 
                                            durations[bidx_i:bidx_j],
                                            sliders[bidx_i:bidx_j],
                                            time_ml[bidx_i:bidx_j],
                                            pred1_conf[bidx_i:bidx_j],
                                            pred2_conf[bidx_i:bidx_j],
                                            rel_conf[bidx_i:bidx_j]
                                            )
                
                total_loss += curr_loss.detach()
                bidx_i = bidx_j
                bidx_j = bidx_i + self.predict_loss_batch_size

                if bidx_j >= total_obs:
                    words = [p for p,q,r in data_x[bidx_i:bidx_j]]
                    spans = [q for p,q,r in data_x[bidx_i:bidx_j]]
                    roots = [r for p,q,r in data_x[bidx_i:bidx_j]]
                    model_out  = self._model(words, spans, roots)   
                    curr_loss = self._custom_temporal_loss(model_out, 
                                                durations[bidx_i:bidx_j],
                                                sliders[bidx_i:bidx_j],
                                                time_ml[bidx_i:bidx_j],
                                                pred1_conf[bidx_i:bidx_j],
                                                pred2_conf[bidx_i:bidx_j],
                                                rel_conf[bidx_i:bidx_j])
                    total_loss += curr_loss.detach()
        
        return total_loss/math.ceil(total_obs/self.predict_loss_batch_size)
        
    def predict(self, data_x, data_y):
        '''
        Predict duration and coarse-grained relations
        '''
        # Turn on evaluation mode which disables dropout.
        self._model.eval()
        
        durations = [x for x,y,z,p,q,r in data_y]
        sliders = [y for x,y,z,p,q,r in data_y]
        time_ml = [z for x,y,z,p,q,r in data_y]
        
        with torch.no_grad():  
            bidx_i = 0
            bidx_j = self.predict_batch_size
            total_obs = len(data_x)
            p1_dur_yhat = torch.zeros(total_obs, 11).to(self.device)
            p2_dur_yhat = torch.zeros(total_obs, 11).to(self.device)
            coarse_yhat = torch.zeros(total_obs, 7).to(self.device)

            while bidx_j < total_obs:
                words = [p for p,q,r in data_x[bidx_i:bidx_j]]
                spans = [q for p,q,r in data_x[bidx_i:bidx_j]]
                roots = [r for p,q,r in data_x[bidx_i:bidx_j]]
                predicts = self._model(words,spans, roots) 
                p1_dur_yhat[bidx_i:bidx_j] = predicts[0]
                p2_dur_yhat[bidx_i:bidx_j] = predicts[1]
                coarse_yhat[bidx_i:bidx_j] = predicts[3]
                
                bidx_i = bidx_j
                bidx_j = bidx_i + self.predict_batch_size

                if bidx_j >= total_obs:
                    words = [p for p,q,r in data_x[bidx_i:bidx_j]]
                    spans = [q for p,q,r in data_x[bidx_i:bidx_j]]
                    roots = [r for p,q,r in data_x[bidx_i:bidx_j]]
                    predicts = self._model(words,spans, roots) 
                    p1_dur_yhat[bidx_i:bidx_j] = predicts[0]
                    p2_dur_yhat[bidx_i:bidx_j] = predicts[1]
                    coarse_yhat[bidx_i:bidx_j] = predicts[3]
            
            p1_dur_yhat = F.softmax(p1_dur_yhat, dim=1)
            p2_dur_yhat = F.softmax(p2_dur_yhat, dim=1)
            coarse_yhat = F.softmax(coarse_yhat, dim=1)

            _ , p1_dur_yhat =  p1_dur_yhat.max(1)
            _ , p2_dur_yhat =  p2_dur_yhat.max(1)
            _ , coarse_yhat =  coarse_yhat.max(1)
            
        self._print_metrics(durations, time_ml, p1_dur_yhat.detach().cpu().numpy(),
                                                p2_dur_yhat.detach().cpu().numpy(),
                                                coarse_yhat.detach().cpu().numpy())

        return p1_dur_yhat.detach(), p2_dur_yhat.detach(), coarse_yhat.detach()

    def _print_metrics(self, durations, time_ml, p1_dur_yhat, p2_dur_yhat, coarse_yhat):

        pred1_durs = [p for p,q in durations]
        pred2_durs = [q for p,q in durations]

        p1_corr = spearmanr(pred1_durs, p1_dur_yhat)
        p2_corr = spearmanr(pred2_durs, p2_dur_yhat)
        print("Spearman Correlations:Pred1 dur: {:06.4f}, Pred2 dur:{:06.4f}".format(p1_corr[0], p2_corr[0]))
        print("P-values: Pred1 dur: {:06.4f}, Pred2 dur:{:06.4f}".format(p1_corr[1], p2_corr[1]))

        for var in ['micro', 'macro']:
            precision = precision_score(time_ml, coarse_yhat, average=var)
            recall = recall_score(time_ml, coarse_yhat, average=var)
            f1 = (2*precision*recall)/(precision+recall)
            print("{} : Precision: {:06.4f},  Recall:{:06.4f}, F1: {:06.4f}".format(var, precision, recall, f1))


        