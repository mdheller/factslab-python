from torch.nn import Module, Linear
from torch.nn import MSELoss, L1Loss, SmoothL1Loss, CrossEntropyLoss
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error as mse, accuracy_score as acc, f1_score as f1
import torch
import torch.nn.functional as F
from scipy.stats import mode
from collections import defaultdict
from functools import partial
from os.path import expanduser


class MLPRegression(Module):
    def __init__(self, attributes, embeddings, embedding_dim=1024,
                 output_size=1, layers=2, device=False, batch_first=True,
                 attention="None"):
        '''Define functional components of network with trainable params'''
        super(MLPRegression, self).__init__()

        # set model hyperparameters
        self.device = device
        self.layers = layers
        self.batch_first = batch_first
        self.embedding_dim = embedding_dim
        self.input_size = int(self.embedding_dim / 4)
        self.output_size = output_size
        self.attributes = attributes
        self.attention = attention
        # Initialise ELMO embeddings
        self.embeddings = embeddings

        # Initialise regression layers
        self._init_regression()

        # Initialise attention
        self._init_attention(self.attention)

    def _init_regression(self):
        '''
            Define the linear maps
        '''

        self.embed_lin_map_lower = Linear(int(self.embedding_dim), int(self.embedding_dim / 4)).to(self.device)
        self.embed_lin_map_mid = Linear(int(self.embedding_dim), int(self.embedding_dim / 4), bias=False).to(self.device)
        self.embed_lin_map_top = Linear(int(self.embedding_dim), int(self.embedding_dim / 4), bias=False).to(self.device)

        self.lin_maps = {}
        for attr in self.attributes:
            self.lin_maps[attr] = []
        for attr in self.attributes:
            last_size = self.input_size
            for i in range(1, self.layers + 1):
                out_size = int(last_size / (i * 4))
                linmap = Linear(last_size, out_size)
                linmap = linmap.to(self.device)
                self.lin_maps[attr].append(linmap)
                varname = '_linear_map' + attr + str(i)
                MLPRegression.__setattr__(self, varname, self.lin_maps[attr][-1])
                last_size = out_size
            linmap = Linear(last_size, self.output_size)
            linmap = linmap.to(self.device)
            self.lin_maps[attr].append(linmap)
            varname = '_linear_map' + attr + str(self.layers + 1)
            MLPRegression.__setattr__(self, varname, self.lin_maps[attr][-1])

        # self.softmax = torch.nn.Softmax(dim=0)
        # self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        # self.Dropout = Dropout()

    def _regression_nonlinearity(self, x):
        return F.relu(x)

    def _init_attention(self, attention_type):
        '''
            Initialises the attention map vector/matrix

            Takes attention_type-Span, Sentence, Span-param, Sentence-param
            as a parameter to decide the size of the attention matrix
        '''
        if self.attention == "None":
            pass
        else:
            if self.attention == "Span" or self.attention == "Sentence":
                # att_map = torch.empty(1, self.input_size)
                self.attention_map = Linear(self.input_size, 1, bias=False).to(self.device)
            elif self.attention == "Span-param" or self.attention == "Sentence-param":
                self.attention_map = Linear(self.input_size, self.input_size).to(self.device)
            # Intialise weights using Xavier method
            # torch.nn.init.xavier_uniform_(att_map)
            # self.attention_map = Parameter(att_map)
            # self.attention_map.to(self.device)

    def _choose_tokens(self, batch, lengths):
        # Index of the last output for each sequence
        idx = (lengths).view(-1, 1).expand(batch.size(0), batch.size(2)).unsqueeze(1)
        return batch.gather(1, idx).squeeze()

    def _choose_span(self, batch, spans):
        '''
            Extract spans from the given batch
        '''
        batch_size, _, _ = batch.shape
        max_span = max([len(i) for i in spans])
        span_batch = torch.zeros((batch_size, max_span, self.input_size), dtype=torch.float, device=self.device)
        for m in range(batch_size):
            span = spans[m]
            for n in range(len(span)):
                span_batch[m, n, :] = batch[m, span[n], :]

        return span_batch

    def _get_inputs(self, words):
        '''Return ELMO embeddings
            Can be done either as a module, or programmatically

            If done programmatically, the 3 layer representations are concatenated, then mapped to a lower dimension and squashed with tanh
        '''
        raw_embeds, _ = self.embeddings.batch_to_embeddings(words)

        embedded_inputs = torch.tanh(
            self.embed_lin_map_lower(raw_embeds[:, 0, :, :].squeeze()) +
            self.embed_lin_map_mid(raw_embeds[:, 1, :, :].squeeze()) +
            self.embed_lin_map_top(raw_embeds[:, 2, :, :].squeeze()))
        return embedded_inputs

    def _run_attention(self, inputs_embed, tokens, spans):

        batch_size = inputs_embed.shape[0]

        if "Span" in self.attention:
            inputs_for_attention = self._choose_span(inputs_embed, spans)
        else:
            inputs_for_attention = inputs_embed

        if self.attention == "None":
            # No attention. Extract root token
            inputs_for_regression = self._choose_tokens(batch=inputs_for_attention, lengths=tokens)

        elif self.attention == "Span" or self.attention == "Sentence":
            att_raw = self.attention_map(inputs_for_attention)
            att = F.softmax(att_raw, dim=1)
            att = att.view(batch_size, 1, att.shape[1])
            inputs_for_regression = torch.bmm(att, inputs_for_attention)

        elif self.attention == "Span-param" or self.attention == "Sentence-param":
            att_param = torch.tanh(self.attention_map(self._choose_tokens(inputs_embed, tokens)))
            att_raw = torch.bmm(att_param.unsqueeze(1), inputs_for_attention.view(batch_size, self.input_size, inputs_for_attention.shape[1]))
            att = F.softmax(att_raw, dim=1)
            inputs_for_regression = torch.bmm(att, inputs_for_attention)
        return inputs_for_regression.squeeze()

    def _run_regression(self, h_in):
        h_out = {}
        for attr in self.attributes:
            h_out[attr] = h_in
            for i, lin_map in enumerate(self.lin_maps[attr]):
                if i:
                    h_out[attr] = self._regression_nonlinearity(h_out[attr])
                h_out[attr] = lin_map(h_out[attr])
                # h_out[attr] = self.Dropout(h_out[attr])
            h_out[attr] = h_out[attr].squeeze()
        return h_out

    def forward(self, inputs, tokens, spans):
        """Forward propagation of activations"""

        inputs_embed = self._get_inputs(inputs)
        inputs_regression = self._run_attention(inputs_embed, tokens, spans)
        outputs = self._run_regression(inputs_regression)

        return outputs


class MLPTrainer:

    loss_function_map = {"linear": MSELoss,
                         "robust": L1Loss,
                         "robust_smooth": SmoothL1Loss,
                         "multinomial": CrossEntropyLoss}

    def __init__(self, regressiontype="linear",
                 optimizer_class=torch.optim.Adam,
                 device="cpu", attributes=["part"],
                 **kwargs):
        self._regressiontype = regressiontype
        self._optimizer_class = optimizer_class
        self.attributes = attributes
        self._init_kwargs = kwargs
        self._continuous = regressiontype != "multinomial"
        self.device = device

    def _initialize_trainer_regression(self):

        lf_class = self.__class__.loss_function_map[self._regressiontype]
        if self._continuous:
            self._regression = MLPRegression(device=self.device,
                                             attributes=self.attributes,
                                             **self._init_kwargs)
            self._loss_function = lf_class()
        else:
            output_size = np.unique(self._Y[0][self.attributes[0]]).shape[0]
            self._regression = MLPRegression(output_size=output_size,
                                             device=self.device,
                                             attributes=self.attributes,
                                             **self._init_kwargs)
            self._loss_function = lf_class(reduction="none")

        self._regression = self._regression.to(self.device)
        self._loss_function = self._loss_function.to(self.device)

    def fit(self, X, Y, loss_wts, tokens, spans, verbosity, dev, epochs):

        self._X = X
        self._Y = Y
        self.dev_x, self.dev_y, self.dev_tokens, self.dev_spans, self.dev_wts = dev
        self.epochs = epochs

        self._initialize_trainer_regression()

        y_ = []
        loss_trace = []
        targ_trace = {}
        pred_trace = {}
        loss_wts_trace = {}
        dev_early = []
        logsft = torch.nn.LogSoftmax(dim=1)
        for attr in self.attributes:
            targ_trace[attr] = []
            pred_trace[attr] = []
            loss_wts_trace[attr] = []

        parameters = [p for p in self._regression.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters)
        epoch = 0
        while epoch < self.epochs:
            epoch += 1
            counter = 0
            if verbosity != 0:
                print("Epoch:", epoch)
                print("Progress" + "\t Metrics(Unweighted, Weighted)")
            for x, y, tks, sps, wts in zip(self._X, self._Y, tokens, spans, loss_wts):
                optimizer.zero_grad()
                counter += 1
                losses = {}

                tks = torch.tensor(tks, dtype=torch.long, device=self.device)
                for attr in self.attributes:
                    if self._continuous:
                        y[attr] = torch.tensor(y[attr], dtype=torch.float, device=self.device)
                    else:
                        y[attr] = torch.tensor(y[attr], dtype=torch.long, device=self.device)
                    wts[attr] = torch.tensor(wts[attr], dtype=torch.float, device=self.device)
                y_ = self._regression(inputs=x, tokens=tks, spans=sps)

                for attr in self.attributes:
                    losses[attr] = self._loss_function(y_[attr], y[attr])
                    if not self._continuous:
                        losses[attr] = torch.mm(losses[attr].unsqueeze(0), wts[attr].unsqueeze(1)).squeeze() / len(losses[attr])
                loss = sum(losses.values())
                loss.backward()
                optimizer.step()

                for attr in self.attributes:
                    targ_trace[attr] += list(y[attr].detach().cpu().numpy())
                    loss_wts_trace[attr] += list(wts[attr].detach().cpu().numpy())
                    if self._continuous:
                        pred_trace[attr] += list(y_[attr].detach().cpu().numpy())
                    else:
                        pred_trace[attr] += list(torch.max(logsft(y_[attr]), 1)[1].detach().cpu().numpy())
                loss_trace.append(float(loss.data))

                if counter % verbosity == 0 or counter == len(self._X) or counter == 1:
                    progress = "{:.2f}".format((counter / len(self._X)) * 100)
                    dev_metrics = self._print_metric(progress=progress,
                                       loss_trace=loss_trace,
                                       targ_trace=targ_trace,
                                       pred_trace=pred_trace,
                                       loss_wts_trace=loss_wts_trace)
                    y_ = []
                    loss_trace = []
                    for attr in self.attributes:
                        targ_trace[attr] = []
                        pred_trace[attr] = []
                        loss_wts_trace[attr] = []

            # EARLY STOPPING
            # Path = expanduser('~') + "/Desktop/saved_models/"
            # import ipdb; ipdb.set_trace()
            # dev_early.append(sum(dev_metrics.values()))
            # if epoch == 1:
            #     torch.save(self.state_dict(), Path)
            # else:
            #     if dev_early[-1] - dev_early[-2] < 0.1:
            #         break
            #     else:
            #         torch.save(self.state_dict(), Path)

    def _print_metric(self, progress, loss_trace, targ_trace, pred_trace, loss_wts_trace=None):
        # Perform validation run
        dev_preds = self.predict(X=self.dev_x, tokens=self.dev_tokens, spans=self.dev_spans)
        if self._continuous:
            sigdig = 3

            train_r2 = {}
            train_mse = {}
            dev_r2 = {}
            dev_mse = {}
            for attr in self.attributes:
                train_r2[attr] = (np.round(r2_score(targ_trace[attr], pred_trace[attr]), sigdig), np.round(r2_score(targ_trace[attr], pred_trace[attr], sample_weight=loss_wts_trace[attr]), sigdig))
                train_mse[attr] = (np.round(mse(targ_trace[attr], pred_trace[attr]), sigdig), np.round(mse(targ_trace[attr], pred_trace[attr], sample_weight=loss_wts_trace[attr]), sigdig))
                dev_r2[attr] = (np.round(r2_score(self.dev_y[attr], dev_preds[attr]), sigdig), np.round(r2_score(self.dev_y[attr], dev_preds[attr], sample_weight=self.dev_wts[attr]), sigdig))
                dev_mse[attr] = (np.round(mse(self.dev_y[attr], dev_preds[attr]), sigdig), np.round(mse(self.dev_y[attr], dev_preds[attr], sample_weight=self.dev_wts[attr]), sigdig))

            print(progress + "%" + '\t\t Total loss:\t', np.round(np.mean(loss_trace), sigdig), '\n',
              ' \t\t R2 DEV:\t', dev_r2, '\n',
              ' \t\t MSE DEV:\t', dev_mse, '\n',
              ' \t\t R2 TRAIN :\t', train_r2, '\n',
              ' \t\t MSE TRAIN:\t', train_mse, '\n')

            return dev_r2
        else:
            sigdig = 3

            train_f1 = {}
            train_acc = {}
            dev_f1 = {}
            dev_acc = {}
            mode_guess_dev = {}
            mode_guess_train = {}
            for attr in self.attributes:
                train_f1[attr] = (np.round(f1(targ_trace[attr], pred_trace[attr]), sigdig), np.round(f1(targ_trace[attr], pred_trace[attr], sample_weight=loss_wts_trace[attr]), sigdig))
                train_acc[attr] = (np.round(acc(targ_trace[attr], pred_trace[attr]), sigdig), np.round(acc(targ_trace[attr], pred_trace[attr], sample_weight=loss_wts_trace[attr]), sigdig))
                dev_f1[attr] = (np.round(f1(self.dev_y[attr], dev_preds[attr]), sigdig), np.round(f1(self.dev_y[attr], dev_preds[attr], sample_weight=self.dev_wts[attr]), sigdig))
                dev_acc[attr] = (np.round(acc(self.dev_y[attr], dev_preds[attr]), sigdig), np.round(acc(self.dev_y[attr], dev_preds[attr], sample_weight=self.dev_wts[attr]), sigdig))
                mode_dev = mode(self.dev_y[attr])
                mode_guess_dev[attr] = (np.round(acc(self.dev_y[attr], [mode_dev[0] for i in range(len(self.dev_y[attr]))]), sigdig), np.round(acc(self.dev_y[attr], [mode_dev[0] for i in range(len(self.dev_y[attr]))], sample_weight=self.dev_wts[attr]), sigdig))
                mode_train = mode(targ_trace[attr])
                mode_guess_train[attr] = (np.round(acc(targ_trace[attr], [mode_train[0] for i in range(len(targ_trace[attr]))]), sigdig), np.round(acc(targ_trace[attr], [mode_train[0] for i in range(len(targ_trace[attr]))], sample_weight=loss_wts_trace[attr]), sigdig))

            print(progress + "%" + '\t\t Total loss:\t', np.round(np.mean(loss_trace), sigdig), '\n',
              ' \t\t F1 DEV:\t', dev_f1, '\n',
              ' \t\t ACC DEV:\t', dev_acc, '\n',
              ' \t\t MODE DEV:\t', mode_guess_dev, '\n',
              ' \t\t F1 TRAIN :\t', train_f1, '\n',
              ' \t\t ACC TRAIN:\t', train_acc, '\n',
              ' \t\t MODE TRAIN:\t', mode_guess_train, '\n')
            early_f1 = {k: v[0] for k, v in dev_f1.items()}
            return early_f1

    def predict(self, X, tokens, spans):
        """Predict using the MLP regression

        Parameters
        ----------
        X : iterable(iterable(object))
            a matrix of structures (independent variables) with rows
            corresponding to a particular kind of RNN
        """
        predictions = defaultdict(partial(np.ndarray, 0))
        logsft = torch.nn.LogSoftmax(dim=1)
        for x, tokens_, spans_ in zip(X, tokens, spans):
            tokens_ = torch.tensor(tokens_, dtype=torch.long, device=self.device)
            y_dev = self._regression(inputs=x, tokens=tokens_, spans=spans_)
            # import ipdb; ipdb.set_trace()
            for attr in self.attributes:
                if self._continuous:
                    predictions[attr] = np.concatenate([predictions[attr], y_dev[attr].detach().cpu().numpy()])
                else:
                    predictions[attr] = np.concatenate([predictions[attr], torch.max(logsft(y_dev[attr]), 1)[1].detach().cpu().numpy()])
            del y_dev
        return predictions
