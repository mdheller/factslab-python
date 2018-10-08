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
     
     Attention types (default: "normal"):
     1. normal: non-parameterized attention (full or span)
     2. param: parameterised attention (full or span)
     
    '''
    
    def __init__(self, embedding_size=1024,
                attention=True,attention_type="normal",regression_hidden_sizes=[24,16], output_size=1,
                 device=torch.device(type="cpu"), batch_size=16):
        super().__init__()

        self.device = device
        self.batch_size = batch_size
        self.attention_type = attention_type
        self.embedding_size = embedding_size
        
        # initialize MLP layers
        self._initialize_regression(self.embedding_size,
                                    regression_hidden_sizes,
                                    output_size) 
        
        self.attention_map = Parameter(torch.zeros(1,self.embedding_size))

        #Attention map vector copied for mini-batch size
        self.attention_map_copied = torch.zeros((self.batch_size, self.embedding_size))
        self.attention_map_copied[:] = self.attention_map.data
    
    def _run_attention(self, embed_input, return_weights=False):
        '''
        Input: An input tensor with dimension:
             (batch_size x max_sentence_len x embedding_size)

        Output: Weighted embeddding with dimension:
                (batch_size x embedding_size)
        '''
        
        att_raw = torch.bmm(embed_input, self.attention_map_copied[:, :, None])
        att = F.softmax(att_raw.squeeze(), dim=1)
                
        if return_weights:
            return att
        else:
            return torch.bmm(att[:, None, :], embed_input).squeeze()

    def _initialize_regression(self, input_size, hidden_sizes, output_size):
        '''
        Initialise MLP or regression parameters
        '''
        self.linear_maps = nn.ModuleList()

        for h in hidden_sizes:
            linmap = torch.nn.Linear(input_size, h)
            linmap = linmap.to(self.device)
            self.linear_maps.append(linmap)
            input_size = h

        linmap = torch.nn.Linear(input_size, output_size)
        linmap = linmap.to(self.device)
        self.linear_maps.append(linmap)

    def _span_inputs(self, inputs, spans):
        '''
        Extract embeddings for a span in the sentence
        
        For a mini-batch, keeps the length of span equal to the length 
        max span in that batch
        '''
        
        span_lengths = [len(x) for x in spans]
        max_span_len = max(span_lengths)
        
        span_embeds = torch.zeros((self.batch_size, max_span_len, self.embedding_size), 
                                  dtype=torch.float, device=self.device)
        
        for sent_idx in range(self.batch_size):
            m=0
            for span_idx in spans[sent_idx]:
                span_embeds[sent_idx][m] = inputs[sent_idx][span_idx]
                m+=1
                
        return span_embeds
    
    def forward(self, structures, idxs=None):
        '''
        Input: 1. structures: A list of list of words
               2. idxs: A list of list of span indexes 
        
        Inputs are run through an attention layer followed by MLP layers
        '''
        #Create elmo dict
        sentences = structures
        character_ids = batch_to_ids(sentences)
        embeddings_dict = elmo(character_ids)

        #inputs
        inputs = embeddings_dict['elmo_representations'][0].to(self.device)
        inputs = inputs.detach()  #Removes embeddings as a parameter (i.e. no back-prop)
        
        if idxs:
            inputs = self._span_inputs(inputs, idxs)

        #pre-process inputs
        inputs = self._preprocess_inputs(inputs)

        #run attention
        output = self._run_attention(inputs)
        
        #run MLP
        output = self._run_regression(output)

        y_hat = self._postprocess_outputs(output)

        return y_hat

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