from torch.nn import Parameter, Dropout, Module, Linear
from torch.nn import MSELoss, L1Loss, SmoothL1Loss, CrossEntropyLoss
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error as mse
from allennlp.modules.elmo import batch_to_ids
import torch
import torch.nn.functional as F


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
        self.lin_maps = {}
        for attr in self.attributes:
            self.lin_maps[attr] = []
        for attr in self.attributes:
            last_size = self.embedding_dim
            for i in range(1, self.layers + 1):
                out_size = int(last_size / (i * 4))
                # import ipdb; ipdb.set_trace()
                linmap = Linear(last_size, out_size)
                linmap = linmap.to(self.device)
                self.lin_maps[attr].append(linmap)
                varname = '_linear_map' + attr + str(i)
                MLPRegression.__setattr__(self, varname, self.lin_maps[attr][-1])
                last_size = out_size
            linmap = torch.nn.Linear(last_size, self.output_size)
            linmap = linmap.to(self.device)
            self.lin_maps[attr].append(linmap)
            varname = '_linear_map' + attr + str(self.layers)
            MLPRegression.__setattr__(self, varname, self.lin_maps[attr][-1])

        self.softmax = torch.nn.Softmax(dim=0)
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.Dropout = Dropout()

    def _regression_nonlinearity(self, x):
        return F.relu(x)

    def _init_attention(self, attention_type):
        if self.attention == "None":
            pass
        elif self.attention == "Span" or self.attention == "Sentence":
            self.attention_map = Parameter(torch.zeros(1, self.embedding_dim))
        elif self.attention == "Span-param" or self.attention == "Sentence-param":
            self.attention_map = Parameter(torch.zeros(self.embedding_dim, self.embedding_dim))

    def _choose_tokens(self, batch, lengths):
        # Index of the last output for each sequence
        idx = (lengths).view(-1, 1).expand(batch.size(0), batch.size(2)).unsqueeze(1)
        return batch.gather(1, idx).squeeze()

    def _get_inputs(self, words):
        '''Return ELMO embeddings'''
        indices = batch_to_ids(words)
        indices = torch.tensor(indices, dtype=torch.long, device=self.device)
        embedded_inputs = self.embeddings(indices)['elmo_representations'][0]
        return embedded_inputs

    def _run_attention(self, inputs_embed, tokens):

        batch_size = inputs_embed.shape[0]
        if self.attention == "None":
            # No attention. Extract root token
            inputs_for_regression = self._choose_tokens(batch=inputs_embed,
                                                        lengths=tokens)
        elif self.attention == "Span" or self.attention == "Sentence":
            att_map = self.attention_map.repeat(batch_size, 1)
            att_raw = torch.bmm(inputs_embed, att_map)
            att = F.softmax(att_raw, dim=0)
            inputs_for_regression = torch.bmm(att, inputs_embed)
        elif self.attention == "Span-param" or self.attention == "Sentence-param":
            att_map = self.attention_map.repeat(batch_size, 1)
            att_param = torch.bmm(inputs_embed, att_map)
            att_raw = torch.bmm(att_param, self._choose_tokens(inputs_embed, tokens))
            att = F.softmax(att_raw, dim=0)
            inputs_for_regression = torch.bmm(att, inputs_embed)
        return inputs_for_regression

    def _run_regression(self, h_in):
        h_out = {}
        for attr in self.attributes:
            h_out[attr] = h_in
            for i, lin_map in enumerate(self.lin_maps[attr]):
                if i:
                    h_out[attr] = self._regression_nonlinearity(h_out[attr])
                h_out[attr] = lin_map(h_out[attr])
            h_out[attr] = h_out[attr].squeeze()
        return h_out

    def forward(self, inputs, tokens, attention="None"):
        """Forward propagation of activations"""

        inputs_embed = self._get_inputs(inputs)
        tokens = torch.tensor(tokens, dtype=torch.long, device=self.device)
        inputs_regression = self._run_attention(inputs_embed, tokens)
        outputs = self._run_regression(inputs_regression)

        return outputs


class MLPTrainer:

    loss_function_map = {"linear": MSELoss,
                         "robust": L1Loss,
                         "robust_smooth": SmoothL1Loss,
                         "multinomial": CrossEntropyLoss}

    def __init__(self, regressiontype="linear",
                 optimizer_class=torch.optim.Adam,
                 device="cpu", epochs=1, attributes=["part"],
                 **kwargs):
        self._regressiontype = regressiontype
        self._optimizer_class = optimizer_class
        self.epochs = epochs
        self.attributes = attributes
        self._init_kwargs = kwargs
        self._continuous = regressiontype != "multinomial"
        self.device = device
        self.epochs = epochs

    def _initialize_trainer_regression(self):
        if self._continuous:
            self._regression = MLPRegression(device=self.device,
                                             attributes=self.attributes,
                                             **self._init_kwargs)
        else:
            output_size = np.unique(self._Y[self.attributes[0]]).shape[0]
            self._regression = MLPRegression(output_size=output_size,
                                             device=self.device,
                                             attributes=self.attributes,
                                             **self._init_kwargs)

        lf_class = self.__class__.loss_function_map[self._regressiontype]
        self._loss_function = lf_class()

        self._regression = self._regression.to(self.device)
        self._loss_function = self._loss_function.to(self.device)

    def fit(self, X, Y, tokens, verbosity, dev):

        self._initialize_trainer_regression()
        self._X = X
        self._Y = Y
        self.dev_x, self.dev_y, self.dev_tokens = dev

        y_ = []
        loss_trace = []
        targ_trace = {}
        pred_trace = {}
        for attr in self.attributes:
            targ_trace[attr] = []
            pred_trace[attr] = []

        parameters = [p for p in self._regression.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters)
        epoch = 0
        while epoch < self.epochs:
            epoch += 1
            counter = 0
            if verbosity != 0:
                print("Epoch:", epoch)
                print("Progress" + "\t Metrics")
            for x, y, tks in zip(self._X, self._Y, tokens):
                optimizer.zero_grad()
                counter += 1
                losses = {}
                # import ipdb; ipdb.set_trace()
                tks = torch.tensor(tks, dtype=torch.long, device=self.device)
                for attr in self.attributes:
                    y[attr] = torch.tensor(y[attr], dtype=torch.float, device=self.device)

                y_ = self._regression(inputs=x, tokens=tks)

                for attr in self.attributes:
                    losses[attr] = self._loss_function(y_[attr], y[attr])

                loss = sum(losses.values())
                loss.backward()
                optimizer.step()

                for attr in self.attributes:
                    targ_trace[attr] += list(y[attr].detach().numpy())
                    pred_trace[attr] += list(y_[attr].detach().numpy())
                loss_trace.append(float(loss.data))

                if counter % verbosity == 0:
                    progress = "{:.4f}".format((counter / len(self._X)) * 100)
                    self._print_metric(progress=progress,
                                       loss_trace=loss_trace,
                                       targ_trace=targ_trace,
                                       pred_trace=pred_trace)
                    y_ = []
                    loss_trace = []
                    for attr in self.attributes:
                        targ_trace[attr] = []
                        pred_trace[attr] = []

    def _print_metric(self, progress, loss_trace, targ_trace, pred_trace, loss_wts_trace=None):

        # Perform validation run
        dev_preds = self.predict(X=self.dev_x, tokens=self.dev_tokens)
        import ipdb; ipdb.set_trace()
        if self._continuous:
            sigdig = 3

            train_r2 = {}
            train_mse = {}
            dev_r2 = {}
            dev_mse = {}

            for attr in self.attributes:
                train_r2[attr] = r2_score(targ_trace[attr], pred_trace[attr])
                train_mse[attr] = mse(targ_trace[attr], pred_trace[attr])
                dev_r2[attr] = r2_score(self.dev_y[attr], dev_preds[attr])
                dev_mse[attr] = mse(self.dev_y[attr], dev_preds[attr])
                # Reduce to sig digits
                train_r2[attr] = np.round(train_r2[attr])
                train_mse[attr] = np.round(train_mse[attr])
                dev_r2[attr] = np.round(dev_r2[attr])
                dev_mse[attr] = np.round(dev_mse[attr])

            print(progress + "%" + '\t\t Total loss:\t', np.round(np.mean(loss_trace), sigdig), '\n',
              ' \t\t R2 on dev:\t', dev_r2, '\n',
              ' \t\t R2 on train :\t', train_r2, '\n',
              ' \t\t MSE on dev:\t', dev_mse,
              ' \t\t MSE on train:\t', train_mse)

    def predict(self, X, tokens):
        """Predict using the LSTM regression

        Parameters
        ----------
        X : iterable(iterable(object))
            a matrix of structures (independent variables) with rows
            corresponding to a particular kind of RNN
        """
        predictions = []
        for x, tokens_ in zip(X, tokens):
            tokens_ = torch.tensor(tokens_, dtype=torch.long, device=self.device)
            predictions.append(self._regression(inputs=x, tokens=tokens_))

        predictions_ = {}
        for attr in self.attributes:
            predictions_[attr] = np.concatenate(np.array([predictions[i][attr].detach().numpy() for i in range(len(predictions))]))
        return predictions_
