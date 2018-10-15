import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.nn import Parameter
from torch.nn import MSELoss, L1Loss, SmoothL1Loss, CrossEntropyLoss


import pandas as pd
import numpy as np

from collections import Iterable, defaultdict

from allennlp.modules.elmo import Elmo, batch_to_ids

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

elmo = Elmo(options_file, weight_file, 1, dropout=0, requires_grad=False)  #using 1 layer of representation

class Attention_mlp(torch.nn.Module):
    
    '''
     A class to run attention models on ELMO word embeddings stacked up with MLP layers
     
     #Predicate attention type:
     1. None i.e. simply get the roots
     2. const-span-attention
     3. param-span-attention

     #Relation type:
     1. concat (simple concatenation of predicate roots)
     2. param-sent-attention 
     
    '''
    
    def __init__(self, embedding_size=1024,
                attention=True, pred_attention_type=None,relation_type="concat", regression_hidden_sizes=[24,16], output_size=1,
                 device=torch.device(type="cpu"), batch_size=4):
        super().__init__()

        self.device = device
        self.batch_size = batch_size
        self.pred_attention_type = pred_attention_type
        self.relation_type = relation_type
        self.embedding_size = embedding_size
        
        # initialize MLP layers
        self._initialize_regression(self.embedding_size,
                                    regression_hidden_sizes,
                                    output_size) 
        #Initialize attention parameters
        self._initialize_attention()

    def _initialize_attention(self):

        #Predicate attention
        if self.pred_attention_type == None:
            pass
        elif self.pred_attention_type == "const-span-attention":
            self.att_map = torch.nn.Linear(self.embedding_size, 1, bias=False)
            
        elif self.pred_attention_type == "param-span-attention":
            self.att_map = torch.nn.Linear(self.embedding_size, self.embedding_size)

        #Relation attention 
        if self.relation_type == "concat":
            pass
        elif self.relation_type == "param-sent-attention":
            self.sent_att_map = torch.nn.Linear(self.embedding_size*2, self.embedding_size)

    def _initialize_regression(self, input_size, hidden_sizes, output_size):
        '''
        Initialise MLP or regression parameters
        '''
        self.linear_maps = nn.ModuleList()

        if self.relation_type == "concat":
            input_size = input_size*2
        else:
            pass

        for h in hidden_sizes:
            linmap = torch.nn.Linear(input_size, h)
            linmap = linmap.to(self.device)
            self.linear_maps.append(linmap)
            input_size = h

        linmap = torch.nn.Linear(input_size, output_size)
        linmap = linmap.to(self.device)
        self.linear_maps.append(linmap)

    def forward(self, structures, spans_idxs, root_idxs):
        '''
        Input: 1. structures: A list of list of words
               2. idxs: A list of list of span indexes 
        
        Inputs are run through an attention layer followed by MLP layers
        '''
        #Create elmo dict
        sentences = structures
        character_ids = batch_to_ids(sentences)
        embeddings_dict = elmo(character_ids)

        #sentence inputs
        inputs = embeddings_dict['elmo_representations'][0].to(self.device)
        inputs = inputs.detach()  #Removes embeddings as a parameter (i.e. no back-prop)

        #pre-process inputs
        inputs = self._preprocess_inputs(inputs)

        #Run attention on inputs based on attention type:
        pred1_out, pred2_out = self._run_pred_attention(inputs, spans_idxs, root_idxs)
            
        #Run through relative_temporal type:
        output = self._run_relative_attention(inputs, pred1_out, pred2_out)

        #####.  MLP.  ######
        #run MLP
        output = self._run_regression(output)

        y_hat = self._postprocess_outputs(output)

        return y_hat

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
        root_inputs = torch.zeros((self.batch_size, self.embedding_size), 
                                    dtype=torch.float, device=self.device)

        for sent_idx in range(self.batch_size):
            root_idx = root_idxs[sent_idx]
            root_inputs[sent_idx] = inputs[sent_idx][root_idx]

        return root_inputs

    def _extract_span_inputs(self, inputs, span_idxs):
        '''
        Extract embeddings for a span in the sentence
        
        For a mini-batch, keeps the length of span equal to the length 
        max span in that batch
        '''
        
        span_lengths = [len(x) for x in span_idxs]
        max_span_len = max(span_lengths)
        
        span_embeds = torch.zeros((self.batch_size, max_span_len, self.embedding_size), 
                                  dtype=torch.float, device=self.device)
        
        for sent_idx in range(self.batch_size):
            m=0
            for span_idx in span_idxs[sent_idx]:
                span_embeds[sent_idx][m] = inputs[sent_idx][span_idx]
                m+=1
                
        return span_embeds

    def _run_pred_attention(self, inputs, spans_idxs, root_idxs, return_weights=False):
        '''
        Input: An input tensor with dimension:
             (batch_size x max_sentence_len x embedding_size)

        Output: pred1, pred2 output emmbeddings after running 
                the corresponding attention types
                
                Shape(pred1_out): (batch_size x embedding_size)
        '''
        #Predicate indexes:
        pred1_r_idxs = [x for x,y in root_idxs]
        pred2_r_idxs = [y for x,y in root_idxs]
        pred1_spans = [x for x,y in spans_idxs]
        pred2_spans = [y for x,y in spans_idxs]

        if self.pred_attention_type==None:
            pred1_out = self._extract_root_inputs(inputs, pred1_r_idxs)
            pred2_out = self._extract_root_inputs(inputs, pred2_r_idxs)

            return pred1_out, pred2_out

        elif self.pred_attention_type == "const-span-attention" :
            pred1_span_inputs = self._extract_span_inputs(inputs, pred1_spans)
            pred2_span_inputs = self._extract_span_inputs(inputs, pred2_spans)

            #Pred1: 
            att_raw = self.att_map(pred1_span_inputs)
            att = F.softmax(att_raw.view(self.batch_size, pred1_span_inputs.shape[1]), dim=1)
            pred1_out = torch.bmm(att[:, None, :], pred1_span_inputs).squeeze()

            #Pred2:
            att_raw = self.att_map(pred2_span_inputs)
            att = F.softmax(att_raw.view(self.batch_size, pred2_span_inputs.shape[1]), dim=1)
            pred2_out = torch.bmm(att[:, None, :], pred2_span_inputs).squeeze()

            return pred1_out, pred2_out


        elif self.pred_attention_type == "param-span-attention":
            pred1_root = self._extract_root_inputs(inputs, pred1_r_idxs)
            pred2_root = self._extract_root_inputs(inputs, pred2_r_idxs)
            pred1_span_inputs = self._extract_span_inputs(inputs, pred1_spans)
            pred2_span_inputs = self._extract_span_inputs(inputs, pred2_spans)

            #Pred1
            att_span = self.att_map(pred1_root)
            att_span = self._regression_nonlinearity(att_span)
            att_raw = torch.bmm(pred1_span_inputs, att_span[:, :, None])
            att = F.softmax(att_raw.view(self.batch_size, pred1_span_inputs.shape[1]), dim=1)
            pred1_out = torch.bmm(att[:, None, :], pred1_span_inputs).squeeze()

            #Pred2
            att_span = self.att_map(pred2_root)
            att_span = self._regression_nonlinearity(att_span)
            att_raw = torch.bmm(pred2_span_inputs, att_span[:, :, None])
            att = F.softmax(att_raw.view(self.batch_size, pred2_span_inputs.shape[1]), dim=1)
            pred2_out = torch.bmm(att[:, None, :], pred2_span_inputs).squeeze()

            return pred1_out, pred2_out


        
    def _run_relative_attention(self, inputs, pred1_out, pred2_out):

        '''
        Inputs:
        1. inputs: Embeddings of the whole sentence
        2. pred1_out: embeddings of pred1 output from attention on pred1
        3. pred2_out: embeddings of pred2 output from attention on pred2

        Output:
        Final layer output combining both the predicates as per the relation_type
        '''
        pred_concat = torch.cat((pred1_out, pred2_out), dim=1)
        
        if self.relation_type == "concat":
            return pred_concat

        else:
            att_span = self.sent_att_map(pred_concat)
            att_span = self._regression_nonlinearity(att_span)
            att_raw = torch.bmm(inputs, att_span[:, :, None])
            att = F.softmax(att_raw.view(self.batch_size, inputs.shape[1]), dim=1)

            out = torch.bmm(att[:, None, :], inputs).squeeze()

            return out


    def _preprocess_inputs(self, inputs):
        """Apply some function(s) to the input embeddings
        This is included to allow for an easy preprocessing hook for
        RNNRegression subclasses. For instance, we might want to
        apply a tanh to the inputs to make them look more like features
        """
        return inputs
 
    def _run_regression(self, h_last):
        for i, linear_map in enumerate(self.linear_maps):
            if i:
                h_last = self._regression_nonlinearity(h_last)
            h_last = linear_map(h_last)
        return h_last
    
    def _postprocess_outputs(self, outputs):
        """Apply some function(s) to the output value(s)"""
        return outputs.squeeze()

    def _regression_nonlinearity(self, x):
        return torch.tanh(x)