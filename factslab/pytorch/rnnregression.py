import numpy as np
import pandas as pd
import torch
import torch.autograd
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import LSTM
from torch.nn import MSELoss, L1Loss, SmoothL1Loss, CrossEntropyLoss
from scipy.special import huber

from random import shuffle
from collections import Iterable
from factslab.utility import partition

from .childsumtreelstm import *

class RNNRegression(torch.nn.Module):
    """Pytorch module for running the out of an RNN through and MLP

    The most basic use-case for this module is to run an RNN (default:
    LSTM) on some inputs, then predict an output using a regression
    (default: linear) on the final state of that LSTM. More complex
    use-cases are also supported:

    Multi-layered regression (the natural generalization of the
    multi-layer perceptron to arbitrary link functions) can be
    implemented by passing an iterable of ints representing hidden
    state sizes for the regression as `regression_hidden_sizes`.

    End-to-end RNN cascades (using the hidden states of one RNN as the
    inputs to another) can be implemented by passing an iterable of
    RNN pytorch module classes as `rnn_classes`.

    The hidden state sizes, number of layers, and bidirectionality for
    each RNN in a cascade can be specified by passing iterables of
    ints, ints, and bools (respectively) to `rnn_hidden_sizes`,
    `num_rnn_layers`, and `bidirectional` (respectively). Note that,
    if an iterable is passed as any of these three parameters, it must
    have the same length as `rnn_classes`. When an iterable isn't
    passed to any one of these parameters, the same value is used for
    all RNNs in the cascade.

    Parameters
    ----------
    embeddings : numpy.array or NoneType
        a vocab-by-embedding dim matrix (e.g. GloVe)
    embedding_size : int or NoneType
        only specify if not passing a pre-trained embedding
    vocab : list(str) or NoneType
        only specify if not passing a pre-trained embedding
    rnn_classes : subclass RNNBase or iterable(subclass RNNBase)
    rnn_hidden_sizes : int or iterable(int)
        the size of the hidden states in each layer of each
        kind of RNN, going from input (RNN hidden state)
        to output; must be same length as rnn_classes
    num_rnn_layers : int or iterable(int)
        must be same length as rnn_hidden_size
    bidirectional : bool or iterable(bool)
        must be same length as rnn_hidden_size
    regression_hidden_sizes : iterable(int)
        the size of the hidden states in each layer of a
        multilayer regression, going from input (RNN hidden state)
        to output
    gpu : bool
        whether to use the gpu
    """


    def __init__(self, embeddings=None, embedding_size=None, vocab=None,
                 rnn_classes=LSTM, rnn_hidden_sizes=300,
                 num_rnn_layers=1, bidirectional=False,
                 regression_hidden_sizes=[], output_size=1, gpu=False):
        super().__init__()

        # set hardware parameters
        self.gpu = gpu
        
        # initialize model
        self._initialize_embeddings(embeddings, vocab)
        self._initialize_rnn(rnn_classes, rnn_hidden_sizes,
                             num_rnn_layers, bidirectional)
        self._initialize_regression(regression_hidden_sizes,
                                    output_size)

    def _homogenize_parameters(self, rnn_classes, rnn_hidden_sizes,
                               num_rnn_layers, bidirectional):
        
        iterables = [p for p in [rnn_classes, rnn_hidden_sizes,
                                 num_rnn_layers, bidirectional]
                     if isinstance(p, Iterable)]
        max_length = max([len(p) for p in iterables]) if iterables else 1

        if not isinstance(rnn_classes, Iterable):
            self.rnn_classes = [rnn_classes]*max_length
        else:
            self.rnn_classes = rnn_classes
            
        if not isinstance(rnn_hidden_sizes, Iterable):
            self.rnn_hidden_sizes = [rnn_hidden_sizes]*max_length
        else:
            self.rnn_hidden_sizes = rnn_hidden_sizes
            
        if not isinstance(num_rnn_layers, Iterable):
            self.num_rnn_layers = [num_rnn_layers]*max_length
        else:
            self.num_rnn_layers = num_rnn_layers
            
        if not isinstance(bidirectional, Iterable):
            self.bidirectional = [bidirectional]*max_length
        else:
            self.bidirectional = bidirectional
            
    def _validate_parameters(self):
        try:
            assert len(self.rnn_classes) == len(self.rnn_hidden_sizes)
            assert len(self.rnn_classes) == len(self.num_rnn_layers)
            assert len(self.rnn_classes) == len(self.bidirectional)
        except AssertionError:
            msg = "rnn_classes, rnn_hidden_sizes,"+\
                  "num_rnn_layers, and bidirectional"+\
                  "must be non-iterable or the same length"
            raise ValueError(msg)
            
    def _initialize_embeddings(self, embeddings, vocab):
        # set embedding hyperparameters
        if embeddings is None:
            self.vocab = vocab
            self.num_embeddings = len(self.vocab)
            self.embedding_size = embedding_size
        else:
            self.num_embeddings, self.embedding_size = embeddings.shape
            self.vocab = embeddings.index

        # define embedding layer
        self.embeddings = torch.nn.Embedding(self.num_embeddings, self.embedding_size,
                                             padding_idx=None, max_norm=None,
                                             norm_type=2, scale_grad_by_freq=False,
                                             sparse=False)

        # copy the embeddings into the embedding layer
        if embeddings is not None:
            embeddings_torch = torch.from_numpy(embeddings.values)
            self.embeddings.weight.data.copy_(embeddings_torch)

        # construct the hash
        self.vocab_hash = {w: i for i, w in enumerate(self.vocab)}
        

    def _initialize_rnn(self, rnn_classes, rnn_hidden_sizes,
                        num_rnn_layers, bidirectional):

        self._homogenize_parameters(rnn_classes, rnn_hidden_sizes,
                                    num_rnn_layers, bidirectional)
        self._validate_parameters()
        
        output_size = self.embedding_size
        self.rnns = []

        params_zipped = zip(self.rnn_classes, self.rnn_hidden_sizes,
                            self.num_rnn_layers, self.bidirectional)
        
        for rnn_class, hsize, lnum, bi in params_zipped:
            input_size = output_size
            rnn = rnn_class(input_size=input_size,
                            hidden_size=hsize,
                            num_layers=lnum,
                            bidirectional=bi)
            rnn = rnn.cuda() if self.gpu else rnn
            self.rnns.append(rnn)
            output_size = hsize*2 if bi else hsize
            
        self.rnn_output_size = output_size

    def _initialize_regression(self, hidden_sizes, output_size):        
        self.linear_maps = []

        last_size = self.rnn_output_size

        for h in hidden_sizes:
            linmap = torch.nn.Linear(last_size, h)
            linmap = linmap.cuda() if self.gpu else linmap
            self.linear_maps.append(linmap)
            last_size = h

        linmap = torch.nn.Linear(last_size, output_size)
        linmap = linmap.cuda() if self.gpu else linmap
        self.linear_maps.append(linmap)
        
    def forward(self, structures):
        """
        Parameters
        ----------
        structures : iterable(object)
           the structures to be used in determining the RNNs
           composition path. Each element must correspond to the
           corresponding RNN in a cascade, or in the case of a trivial
           cascade (a single RNN followed by regression), `structures`
           must be a singleton iterable. When the relevant RNN in a
           cascade is a linear-chain RNN, the structure in the
           corresponding position of this parameter is ignored
        """

        try:
            words = structures[0].words()
        except AttributeError:
            assert all([isinstance(w, str)
                        for w in structures[0]])
            words = structures[0]
        except AssertionError:
            msg = "first structure in sequence must either"+\
                  "implement a words() method or itself be"+\
                  "a sequence of words"
            raise ValueError(msg)
        
        inputs = self._get_inputs(words)

        # passes inputs through tanh before sending them
        # through the tree; done to make the verb reps
        # learned more feature-like
        inputs = self._preprocess_inputs(inputs)
        
        for rnn, structure in zip(self.rnns, structures):            
            if isinstance(rnn, ChildSumTreeLSTM):
                h_all, h_last = rnn(inputs, structure)
            else:
                h_all, h_last = rnn(inputs[:,None,:])

            inputs = h_all.squeeze()
            
        for i, linear_map in enumerate(self.linear_maps):
            if i:
                h_last = self._regression_nonlinearity(h_last)

            h_last = linear_map(h_last)
            
        y_hat = self._postprocess_outputs(h_last)

        return y_hat

    def _regression_nonlinearity(self, x):
        return F.tanh(x)
    
    def _preprocess_inputs(self, inputs):
        """Apply some function(s) to the input embeddings

        This is included to allow for an easy preprocessing hook for 
        RNNRegression subclasses. For instance, we might want to 
        apply a tanh to the inputs to make them look more like features
        """
        return inputs

    def _postprocess_outputs(self, outputs):
        """Apply some function(s) to the output value(s)"""
        return outputs

    
    def _get_inputs(self, words):
        indices = [[self.vocab_hash[w]] for w in words]
        indices = torch.LongTensor(indices)
        indices = indices.cuda() if self.gpu else indices
        indices = Variable(indices)
        
        return self.embeddings(indices).squeeze()

    
    def word_embeddings(self, words=[]):
        embeddings = self._get_inputs(words).data.cpu().numpy()
        return pd.DataFrame(embeddings, index=self.vocab)
    
    
class RNNRegressionTrainer(object):

    loss_function_map = {"linear": MSELoss,
                         "robust": L1Loss,
                         "robust_smooth": SmoothL1Loss,
                         "multinomial": CrossEntropyLoss} 
    
    def __init__(self, regression_type="linear",
                 optimizer_class=torch.optim.Adam, gpu=False, **kwargs):
        self._regression_type = regression_type
        self._optimizer_class = optimizer_class
        self._init_kwargs = kwargs
        
        self._continuous = regression_type != "multinomial"
        
        self.gpu = gpu

    def _initialize_regression(self):
        if self._continuous:
            self._regression = RNNRegression(gpu=self.gpu,
                                             **self._init_kwargs)
        else:
            output_size = np.unique(self._Y).shape[0]
            self._regression = RNNRegression(output_size=output_size,
                                             gpu=self.gpu,
                                             **self._init_kwargs)

        lf_class = self.__class__.loss_function_map[self._regression_type]
        self._loss_function = lf_class()

        if self.gpu:
            self._regression = self._regression.cuda()
            self._loss_function = self._loss_function.cuda()        
        
    def fit(self, X, Y, batch_size=100, verbosity=1, **kwargs):
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

        self._X, self._Y = X, Y

        self._initialize_regression()

        optimizer = self._optimizer_class(self._regression.parameters(),
                                          **kwargs)
        
        if not self._continuous:
            Y_counts = np.bincount(self._Y)
            self._Y_logprob = np.log(Y_counts)-np.log(np.sum(Y_counts))
        
        # each element is of the form ((struct1, struct2, ...),
        #                              target)
        structures_targets = list(zip(zip(*X), Y))

        loss_trace = []
        targ_trace = []
        
        while True:
            losses = []
            
            shuffle(structures_targets)
            part = partition(structures_targets, batch_size)
            
            for i, structs_targs_batch in enumerate(part):

                optimizer.zero_grad()

                for struct, targ in structs_targs_batch:
                    targ_trace.append(targ)

                    if self._continuous:
                        targ = torch.FloatTensor([targ])
                    else:
                        targ = torch.LongTensor([int(targ)])
                        
                    targ = targ.cuda() if self.gpu else targ
                    targ = Variable(targ, requires_grad=False)

                    predicted = self._regression(struct)
                    # predicted = predicted.expand_as(targ)

                    if self._continuous:
                        loss = self._loss_function(predicted, targ)
                    else:
                        loss = self._loss_function(predicted[None,:], targ)

                    losses.append(loss)

                loss = sum(losses)/float(batch_size)
                loss.backward()

                optimizer.step()
                losses = []
                
                loss_trace.append(loss.data[0])

                # TODO: generalize for non-linear regression
                if verbosity and i:
                    if not i % verbosity:
                        self._print_metric(i, loss_trace, targ_trace)
                        loss_trace = []
                        targ_trace = []

    def _print_metric(self, i, loss_trace, targ_trace):

        sigdig = 3
        
        if self._continuous:
            resid_mean = np.mean(loss_trace)

            if self._regression_type == "linear":
                targ_var = np.mean(np.square(np.array(targ_trace)-np.mean(self._Y)))
                r2 = 1. - (resid_mean/targ_var)
                
                print(str(i)+'\t residual variance:\t', np.round(resid_mean, sigdig), '\n',
                      ' \t total variance:\t', np.round(targ_var, sigdig), '\n',
                      ' \t r-squared:\t\t', np.round(r2, sigdig), '\n')
                
            elif self._regression_type == "robust":
                ae = np.abs(targ_trace-np.median(self._Y))
                mae = np.mean(ae)
                pmae = 1. - (resid_mean/mae)

                print(str(i)+'\t residual absolute error:\t', np.round(resid_mean, sigdig), '\n',
                      ' \t total absolute error:\t\t', np.round(mae, sigdig), '\n',
                      ' \t proportion absolute error:\t', np.round(pmae, sigdig), '\n')
                
            elif self._regression_type == "robust_smooth":
                ae = huber(1., targ_trace-np.median(self._Y))
                mae = np.mean(ae)
                pmae = 1. - (resid_mean/mae)

                print(str(i)+'\t residual absolute error:\t', np.round(resid_mean, sigdig), '\n',
                      ' \t total absolute error:\t\t', np.round(mae, sigdig), '\n',
                      ' \t proportion absolute error:\t', np.round(pmae, sigdig), '\n')

        else:
            model_mean_neglogprob = np.mean(loss_trace)
            targ_mean_neglogprob = -np.mean(self._Y_logprob[targ_trace])
            pnlp = 1. - (model_mean_neglogprob/targ_mean_neglogprob)
            
            print(str(i)+'\t residual mean cross entropy:\t', np.round(model_mean_neglogprob, sigdig), '\n',
                  ' \t total mean cross entropy:\t', np.round(targ_mean_neglogprob, sigdig), '\n',
                  ' \t proportion entropy explained:\t', np.round(pnlp, sigdig), '\n')
