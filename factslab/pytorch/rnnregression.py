import numpy as np
import pandas as pd
import torch
import torch.autograd
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn import LSTM
from torch.nn import MSELoss, L1Loss, SmoothL1Loss, CrossEntropyLoss
from scipy.special import huber
import pdb
from random import shuffle
from collections import Iterable
from factslab.utility import partition
from .childsumtreelstm import *
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
import sys


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
    attention : bool
        whether to use attention on the final RNN
    regression_hidden_sizes : iterable(int)
        the size of the hidden states in each layer of a
        multilayer regression, going from input (RNN hidden state)
        to output
    device : torch.device
        device(type="cpu") or device(type="cuda:0")
    """

    def __init__(self, embeddings=None, embedding_size=None, vocab=None,
                 rnn_classes=LSTM, rnn_hidden_sizes=300,
                 num_rnn_layers=1, bidirectional=False, attention=False,
                 regression_hidden_sizes=[], output_size=1,
                 device=torch.device(type="cpu"), batch_size=128):
        super().__init__()

        self.device = device
        self.batch_size = batch_size
        # initialize model
        self._initialize_embeddings(embeddings, vocab)
        self._initialize_rnn(rnn_classes, rnn_hidden_sizes,
                             num_rnn_layers, bidirectional)
        self._initialize_regression(attention,
                                    regression_hidden_sizes,
                                    output_size)

    def _homogenize_parameters(self, rnn_classes, rnn_hidden_sizes,
                               num_rnn_layers, bidirectional):

        iterables = [p for p in [rnn_classes, rnn_hidden_sizes,
                                 num_rnn_layers, bidirectional]
                     if isinstance(p, Iterable)]
        max_length = max([len(p) for p in iterables]) if iterables else 1

        if not isinstance(rnn_classes, Iterable):
            self.rnn_classes = [rnn_classes] * max_length
        else:
            self.rnn_classes = rnn_classes

        if not isinstance(rnn_hidden_sizes, Iterable):
            self.rnn_hidden_sizes = [rnn_hidden_sizes] * max_length
        else:
            self.rnn_hidden_sizes = rnn_hidden_sizes

        if not isinstance(num_rnn_layers, Iterable):
            self.num_rnn_layers = [num_rnn_layers] * max_length
        else:
            self.num_rnn_layers = num_rnn_layers

        if not isinstance(bidirectional, Iterable):
            self.bidirectional = [bidirectional] * max_length
        else:
            self.bidirectional = bidirectional

    def _validate_parameters(self):
        try:
            assert len(self.rnn_classes) == len(self.rnn_hidden_sizes)
            assert len(self.rnn_classes) == len(self.num_rnn_layers)
            assert len(self.rnn_classes) == len(self.bidirectional)
        except AssertionError:
            msg = "rnn_classes, rnn_hidden_sizes," +\
                  "num_rnn_layers, and bidirectional" +\
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
                            bidirectional=bi,
                            batch_first=True)
            rnn = rnn.to(self.device)
            self.rnns.append(rnn)
            output_size = hsize * 2 if bi else hsize

        self.rnn_output_size = output_size
        if self.batch_size > 1:
            self.has_batch_dim = True
        else:
            self.has_batch_dim = False

    def _initialize_regression(self, attention, hidden_sizes, output_size):
        self.linear_maps = []

        last_size = self.rnn_output_size

        self.attention = attention

        if self.attention:
            if self.has_batch_dim:
                self.attention_map = Parameter(torch.zeros(self.batch_size,
                                                           last_size))
            else:
                self.attention_map = Parameter(torch.zeros(last_size))

        for h in hidden_sizes:
            linmap = torch.nn.Linear(last_size, h)
            linmap = linmap.to(self.device)
            self.linear_maps.append(linmap)
            last_size = h

        linmap = torch.nn.Linear(last_size, output_size)
        linmap = linmap.to(self.device)
        self.linear_maps.append(linmap)

    def forward(self, structures, targets):
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
        targets: list
            A list of all the targets in the batch. This will be modified only
            if the rnn_class is LSTM(since the order will be modified
            during padding). Otherwise it is returned as is.
        """

        try:
            words = structures.words()
        except AttributeError:
            # assert all([isinstance(w, str) for w in structures])
            words = structures
        except AssertionError:
            msg = "first structure in sequence must either" +\
                  "implement a words() method or itself be" +\
                  "a sequence of words"
            raise ValueError(msg)
        if isinstance(words[0], list):
            self.has_batch_dim = True
        else:
            self.has_batch_dim = False
        inputs, targets, lengths = self._get_inputs(words, targets)
        inputs = self._preprocess_inputs(inputs)
        h_all, h_last = self._run_rnns(inputs, structures, lengths)

        if self.attention:
            h_last = self._run_attention(h_all)
        else:
            if self.has_batch_dim:
                h_last = self.last_timestep(h_all, lengths)
            else:
                h_last = h_last.view(1, 600)

        h_last = self._run_regression(h_last)

        y_hat = self._postprocess_outputs(h_last)

        return y_hat, targets

    def _run_rnns(self, inputs, structures, lengths):
        '''
            Run desired rnns
        '''
        for rnn, structure in zip(self.rnns, [structures]):
            if isinstance(rnn, ChildSumTreeLSTM):
                h_all, h_last = rnn(inputs, structure)
            elif isinstance(rnn, LSTM) and lengths is not None:
                packed = pack_padded_sequence(inputs, lengths.data, batch_first=True)
                h_all, (h_last, c_last) = rnn(packed)
                h_all, _ = pad_packed_sequence(h_all, batch_first=True)
            else:
                h_all, (h_last, c_last) = rnn(inputs.unsqueeze(0))
            inputs = h_all.squeeze()

        return h_all, h_last

    def _run_attention(self, h_all, return_weights=False):
        if not self.has_batch_dim:
            att_raw = torch.mm(h_all, self.attention_map[:, None])
            att = F.softmax(att_raw.squeeze(), dim=0)

            if return_weights:
                return att
            else:
                return torch.mm(att[None, :], h_all).squeeze()
        else:
            att_raw = torch.bmm(h_all, self.attention_map[:, :, None])
            att = F.softmax(att_raw.squeeze(), dim=0)

            if return_weights:
                return att
            else:
                return torch.bmm(att[:, None, :], h_all).squeeze()

    def _run_regression(self, h_last):
        # Neural davidsonian(simple)
        # pdb.set_trace()
        # h_shared = F.relu(torch.mm(self.attr_shared, h_last.unsqueeze(1)))
        # h = {attr: None for attr in self.attributes}
        # for attr in self.attributes:
        #     h[attr] = torch.mm(torch.transpose(h_shared, 0, 1), self.attr_sp[attr]).squeeze()
        for i, linear_map in enumerate(self.linear_maps):
            if i:
                h_last = self._regression_nonlinearity(h_last)
            h_last = linear_map(h_last)
        return h_last

    def _regression_nonlinearity(self, x):
        return F.tanh(x)

    def _preprocess_inputs(self, inputs):
        """Apply some function(s) to the input embeddings

        This is included to allow for an easy preprocessing hook for
        RNNRegression subclasses. For instance, we might want to
        apply a tanh to the inputs to make them look more like features
        """
        return inputs

    def _pad_inputs(self, data, targets):
        """
            Pad input sequences so that each minibatch has same length
        """
        seq_len = torch.from_numpy(np.array([len(x) for x in data]))
        sorted_seq_len, sorted_idx = seq_len.sort(descending=True)
        sorted_data = torch.zeros((2,), dtype=torch.long, device=self.device)
        sorted_data = sorted_data.new_full((len(data), sorted_seq_len[0]),
                                      fill_value=0)
        sorted_targets = torch.zeros((2,), dtype=torch.long, device=self.device)
        sorted_targets = sorted_targets.new_full((len(targets),), fill_value=0,
                                         dtype=torch.float, device=self.device)
        m = 0
        for x in sorted_idx:
            sorted_data[m][0:len(data[x])] = torch.tensor(data[x], dtype=torch.long)
            sorted_targets[m] = targets[x]
            m += 1

        return sorted_data, sorted_targets, sorted_seq_len

    def _postprocess_outputs(self, outputs):
        """Apply some function(s) to the output value(s)"""
        return outputs.squeeze()

    def last_timestep(self, unpacked, lengths):
        # Index of the last output for each sequence
        idx = (lengths - 1).view(-1, 1).expand(unpacked.size(0), unpacked.size(2)).unsqueeze(1).to(self.device)
        return unpacked.gather(1, idx).squeeze()

    def _get_inputs(self, inputs, targets):
        if self.has_batch_dim:
            indices = []
            for sent in inputs:
                indices.append([self.vocab_hash[word] for word in sent])
            indices, targets, lengths = self._pad_inputs(indices, targets)
            indices = torch.tensor(indices, dtype=torch.long, device=self.device)
            return self.embeddings(indices).squeeze(), targets, lengths
        else:
            indices = [self.vocab_hash[word] for word in inputs]
            indices = torch.tensor(indices, dtype=torch.long,
                                   device=self.device)
            return self.embeddings(indices).squeeze(), targets, None

    def word_embeddings(self, words=[]):
        """Extract the tuned word embeddings

        If an empty list is passed, all word embeddings are returned

        Parameters
        ----------
        words : list(str)
            The words to get the embeddings for

        Returns
        -------
        pandas.DataFrame
        """
        words = words if words else self.vocab
        embeddings = self._get_inputs(words).data.cpu().numpy()

        return pd.DataFrame(embeddings, index=words)

    def attention_weights(self, structures):
        """Compute what the LSTM regression is attending to

        The weights that are returned are only for the structures used
        in the last LSTM. This is because that is the only place that
        attention is implemented - i.e. right befoe passing the LSTM
        outputs to a regression layer.

        Parameters
        ----------
        structures : iterable(iterable(object))
            a matrix of structures (independent variables) with rows
            corresponding to a particular kind of RNN

        Returns
        -------
        pytorch.Tensor
        """
        try:
            assert self.attention
        except AttributeError:
            raise AttributeError('attention not used')

        try:
            words = structures[0].words()
        except AttributeError:
            assert all([isinstance(w, str)
                        for w in structures[0]])
            words = structures[0]
        except AssertionError:
            msg = "first structure in sequence must either" +\
                  "implement a words() method or itself be" +\
                  "a sequence of words"
            raise ValueError(msg)

        inputs = self._get_inputs(words)
        inputs = self._preprocess_inputs(inputs)

        h_all, h_last = self._run_rnns(inputs, structures)

        return self._run_attention(h_all, return_weights=True)


class RNNRegressionTrainer(object):

    loss_function_map = {"linear": MSELoss,
                         "robust": L1Loss,
                         "robust_smooth": SmoothL1Loss,
                         "multinomial": CrossEntropyLoss}

    def __init__(self, regression_type="linear",
                 optimizer_class=torch.optim.Adam,
                 rnn_classes=LSTM,
                 device="cpu", epochs=1,
                 **kwargs):
        self._regression_type = regression_type
        self._optimizer_class = optimizer_class
        self.epochs = epochs
        self._init_kwargs = kwargs
        self.rnn_classes = rnn_classes
        self._continuous = regression_type != "multinomial"
        self.device = device

    def _initialize_trainer_regression(self):
        if self._continuous:
            self._regression = RNNRegression(device=self.device,
                                             rnn_classes=self.rnn_classes,
                                             **self._init_kwargs)
        else:
            output_size = np.unique(self._Y[0]).shape[0]
            self._regression = RNNRegression(output_size=output_size,
                                             device=self.device,
                                             rnn_classes=self.rnn_classes,
                                             **self._init_kwargs)

        lf_class = self.__class__.loss_function_map[self._regression_type]
        self._loss_function = lf_class(reduce=False)

        self._regression = self._regression.to(self.device)
        self._loss_function = self._loss_function.to(self.device)

    def fit(self, X, Y, batch_size=100, verbosity=1, loss_weights=None, **kwargs):
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

        self._initialize_trainer_regression()

        optimizer = self._optimizer_class(self._regression.parameters(),
                                          **kwargs)

        if not self._continuous:
            Y_counts = np.bincount([y for batch in self._Y for y in batch])
            self._Y_logprob = np.log(Y_counts) - np.log(np.sum(Y_counts))

        # each element is of the form ((struct1, struct2, ...),
        #                              target)
        structures_targets = list(zip(self._X, self._Y, loss_weights))
        loss_trace = []
        targ_trace = np.array([])
        epoch = 0
        while epoch < self.epochs:
            epoch += 1
            print("Epoch:", epoch, "\n")
            print("Progress" + "\t Metrics")
            losses = []

            shuffle(structures_targets)
            total = len(self._Y)
            # part = partition(structures_targets, batch_size)
            for i, structs_targs_batch in enumerate(structures_targets):
                optimizer.zero_grad()
                if self.rnn_classes == LSTM:
                    structs, targs, loss_wts = structs_targs_batch
                    loss_wts = torch.tensor(loss_wts, dtype=torch.float, device=self.device)
                    targ_trace = np.append(targ_trace, targs)

                    if self._continuous:
                        targs = torch.tensor(targs, dtype=torch.long, device=self.device)
                    else:
                        targs = torch.tensor(targs, dtype=torch.long, device=self.device)

                    predicted, targs = self._regression(structs, targs)
                    targs = torch.tensor(targs, dtype=torch.long, device=self.device)
                    loss = self._loss_function(predicted, targs)
                    losses.append((torch.mm(loss_wts[None, :], loss[:, None])) / len(loss))
                else:
                    # pdb.set_trace()
                    structs_targs = list(zip(structs_targs_batch[0],
                                             structs_targs_batch[1],
                                             structs_targs_batch[2]))
                    for struct, targ, loss_wt in structs_targs:
                        targ_trace = np.append(targ_trace, targ)

                        if self._continuous:
                            targ = torch.tensor([targ], dtype=torch.float)
                        else:
                            targ = torch.tensor([int(targ)], dtype=torch.long)

                        targ = targ.to(self.device)
                        predicted, targ = self._regression(struct, targ)
                        if self._continuous:
                            loss = loss_wt * self._loss_function(predicted, targ)
                        else:
                            loss = loss_wt * self._loss_function(predicted[None, :], targ)

                        losses.append(loss)

                loss = sum(losses) / len(losses)
                loss.backward()

                optimizer.step()
                losses = []
                loss_trace.append(loss.item())

                # TODO: generalize for non-linear regression
                if verbosity:
                    if not (i + 1) % verbosity:
                        progress = "{:.4f}".format(((i) / total) * 100)
                        self._print_metric(progress, loss_trace, targ_trace, loss_wts)
                        loss_trace = []
                        targ_trace = np.array([])

        torch.save(self._regression.state_dict(), 'genregression.dat')

    def _print_metric(self, progress, loss_trace, targ_trace, loss_wts):

        sigdig = 3
        Y_flat = [y for batch in self._Y for y in batch]
        if self._continuous:
            resid_mean = np.mean(loss_trace)

            if self._regression_type == "linear":
                targ_var = np.mean(np.square(np.array(targ_trace) - np.mean(Y_flat)))
                r2 = 1. - (resid_mean / targ_var)
                print(progress + "%" + '\t\t residual variance:\t', np.round(resid_mean, sigdig), '\n',
                      ' \t\t total variance:\t', np.round(targ_var, sigdig), '\n',
                      ' \t\t r-squared:\t\t', np.round(r2, sigdig), '\n')

            elif self._regression_type == "robust":
                ae = np.abs(targ_trace - np.median(Y_flat))
                mae = np.mean(ae)
                pmae = 1. - (resid_mean / mae)

                print(progress + "%" + '\t\t residual absolute error:\t', np.round(resid_mean, sigdig), '\n',
                      ' \t\t total absolute error:\t\t', np.round(mae, sigdig), '\n',
                      ' \t\t proportion absolute error:\t', np.round(pmae, sigdig), '\n')

            elif self._regression_type == "robust_smooth":
                ae = huber(1., targ_trace - np.median(Y_flat))
                mae = np.mean(ae)
                pmae = 1. - (resid_mean / mae)

                print(progress + "%" + '\t\t residual absolute error:\t', np.round(resid_mean, sigdig), '\n',
                      ' \t\t total absolute error:\t\t', np.round(mae, sigdig), '\n',
                      ' \t\t proportion absolute error:\t', np.round(pmae, sigdig), '\n')

        else:
            model_mean_neglogprob = np.mean(loss_trace)
            targ_mean_neglogprob = -np.mean([a * b for a, b in zip(loss_wts, [self._Y_logprob[int(x)] for x in targ_trace])])
            pnlp = 1. - (model_mean_neglogprob / targ_mean_neglogprob)

            print(progress + "%" + '\t\t residual mean cross entropy:\t', np.round(model_mean_neglogprob, sigdig), '\n',
                  ' \t\t total mean cross entropy:\t', np.round(targ_mean_neglogprob, sigdig), '\n',
                  ' \t\t proportion entropy explained:\t', np.round(pnlp, sigdig), '\n')

    def predict(self, X, Y):
        """Predict using the LSTM regression

        Parameters
        ----------
        X : iterable(iterable(object))
            a matrix of structures (independent variables) with rows
            corresponding to a particular kind of RNN
        """

        predictions, targets = self._regression(X, Y)

        if self._continuous:
            return predictions, targets
        else:
            return predictions, targets

    def attention_weights(self, X):
        """Compute what the LSTM regression is attending to

        Parameters
        ----------
        X : iterable(iterable(object))
            a matrix of structures (independent variables) with rows
            corresponding to a particular kind of RNN

        Returns
        -------
        list(np.array)
        """
        attention = [self._regression.attention_weights(struct)
                     for struct in zip(*X)]
        return [a.data.cpu().numpy() for a in attention]

    def word_embeddings(self, words=[]):
        """Extract the tuned word embeddings

        If an empty list is passed, all word embeddings are returned

        Parameters
        ----------
        words : list(str)
            The words to get the embeddings for

        Returns
        -------
        pandas.DataFrame
        """
        return self._regression.word_embeddings(words)
