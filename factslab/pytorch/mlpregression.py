from torch.nn import Module, Linear, ModuleDict, ModuleList
from torch.nn import MSELoss, L1Loss, SmoothL1Loss, CrossEntropyLoss, NLLLoss
import numpy as np
from sklearn.metrics import accuracy_score as acc, f1_score as f1, precision_score as prec, recall_score as rec, r2_score as r2, mean_squared_error as mse
import torch
import torch.nn.functional as F
from scipy.stats import mode
from collections import defaultdict
from functools import partial
from allennlp.commands.elmo import ElmoEmbedder
from os.path import expanduser
from tqdm import tqdm


class MLPRegression(Module):
    def __init__(self, embed_params, attention_type, all_attrs, device="cpu",
                 embedding_dim=1024, output_size=3, layers=2):
        '''
            Super class for training
        '''
        super(MLPRegression, self).__init__()

        # Set model constants and embeddings
        self.device = device
        self.layers = layers
        self.embedding_dim = embedding_dim
        self.reduced_embedding_dim = int(self.embedding_dim / 4)
        self.output_size = output_size
        self.attention_type = attention_type
        self.all_attributes = all_attrs
        # Initialise embeddings
        self.embeddings = self._init_embeddings(embed_params)

        # Initialise regression layers and parameters
        self._init_regression()

        # Initialise attention parameters
        self._init_attention()

    def _init_embeddings(self, elmo_params):
        options_file = elmo_params[0]
        weight_file = elmo_params[1]
        elmo = ElmoEmbedder(options_file, weight_file, cuda_device=0)
        return elmo

    def _init_regression(self):
        '''
            Define the linear maps
        '''

        # ELMO tuning parameters
        self.embed_lin_map_lower = Linear(self.embedding_dim, self.reduced_embedding_dim)
        self.embed_lin_map_mid = Linear(self.embedding_dim, self.reduced_embedding_dim, bias=False)
        self.embed_lin_map_top = Linear(self.embedding_dim, self.reduced_embedding_dim, bias=False)

        # Output regression parameters
        self.lin_maps = ModuleDict({'arg': ModuleList([]),
                                    'pred': ModuleList([])})
        regr_input = {}
        for prot in self.all_attributes.keys():
            regr_input[prot] = self.reduced_embedding_dim
            # Handle varying size of dimension depending on representation
            if self.attention_type[prot]['repr'] == "root":
                if self.attention_type[prot]['context'] != "none":
                    regr_input[prot] *= 2
            else:
                if self.attention_type[prot]['context'] == "none":
                    regr_input[prot] *= 2
                else:
                    regr_input[prot] *= 3
            for attr in self.all_attributes[prot]:
                self.lin_maps[prot] = []
            for attr in self.all_attributes[prot]:
                last_size = regr_input[prot]
                for i in range(1, self.layers + 1):
                    out_size = int(last_size / (i * 4))
                    linmap = Linear(last_size, out_size)
                    self.lin_maps[prot][attr].append(linmap)
                    varname = '_linear_map' + prot + attr + str(i)
                    MLPRegression.__setattr__(self, varname, self.lin_maps[prot][attr][-1])
                    last_size = out_size
                linmap = Linear(last_size, self.output_size)
                self.lin_maps[prot][attr].append(linmap)
                varname = '_linear_map' + prot + attr + str(self.layers + 1)
                MLPRegression.__setattr__(self, varname, self.lin_maps[prot][attr][-1])

    def _regression_nonlinearity(self, x):
        return F.relu(x)

    def _init_attention(self):
        '''
            Initialises the attention map vector/matrix

            Takes attention_type-Span, Sentence, Span-param, Sentence-param
            as a parameter to decide the size of the attention matrix
        '''

        self.attention_map_repr = {}
        self.attention_map_context = {}
        for prot in self.attention_type.keys():
            # Token representation
            if self.attention_type[prot]['repr'] == "span":
                self.attention_map_repr[prot] = Linear(self.reduced_embedding_dim, 1, bias=False)
                varname = '_att_map_' + prot + "_repr"
                MLPRegression.__setattr__(self, varname, self.attention_map_repr[prot])
            elif self.attention_type[prot]['repr'] == "param":
                self.attention_map_repr[prot] = Linear(self.reduced_embedding_dim, self.reduced_embedding_dim)
                varname = '_att_map_' + prot + "_repr"
                MLPRegression.__setattr__(self, varname, self.attention_map_repr[prot])

            # Context attention
            if self.attention_type[prot]['repr'] in ["span", "param"]:
                repr_dim = self.reduced_embedding_dim
            else:
                repr_dim = self.reduced_embedding_dim
            if self.attention_type[prot]['context'] == "david":
                if prot == "pred":
                    self.attention_map_context[prot] = Linear(repr_dim, repr_dim)
                    varname = '_att_map_' + prot + "_context_david"
                    MLPRegression.__setattr__(self, varname, self.attention_map_context[prot])
            elif self.attention_type[prot]['context'] == "param":
                self.attention_map_context[prot] = Linear(repr_dim, repr_dim)
                varname = '_att_map_' + prot + "_context_param"
                MLPRegression.__setattr__(self, varname, self.attention_map_context[prot])

    def _choose_tokens(self, batch, lengths):
        # Index of the last output for each sequence
        import ipdb; ipdb.set_trace()  # breakpoint b59781ab //
        idx = (lengths).view(-1, 1).expand(batch.size(0), batch.size(2)).unsqueeze(1)
        return batch.gather(1, idx).squeeze()

    def _choose_span_context(self, batch, spans):
        '''
            Extract spans/contexts from the given batch
        '''
        batch_size, _, _ = batch.shape
        max_span = max([len(i) for i in spans])
        span_batch = torch.zeros((batch_size, max_span, self.reduced_embedding_dim), dtype=torch.float, device=self.device)
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
        raw_embeds, masks = self.embeddings.batch_to_embeddings(words)
        embedded_inputs = torch.tanh(
            self.embed_lin_map_lower(raw_embeds[:, 0, :, :].squeeze()) +
            self.embed_lin_map_mid(raw_embeds[:, 1, :, :].squeeze()) +
            self.embed_lin_map_top(raw_embeds[:, 2, :, :].squeeze()))
        masked_embedded_inputs = embedded_inputs * masks
        return masked_embedded_inputs

    def _get_representation(self, prot, embeddings, tokens, spans):
        '''
            returns the representation required from arguments passed by
            running attention based on arguments passed
        '''

        batch_size = embeddings.shape[0]
        # Get token(pred/arg) representation
        rep_type = self.attention_type[prot]['repr']

        if rep_type == "root":
            token_rep = self._choose_tokens(batch=embeddings, lengths=tokens)
        elif rep_type == "span":
            token_rep_raw = self._choose_span_context(embeddings, spans)
            att_raw = self.attention_map_repr[prot](token_rep_raw)
            att = F.softmax(att_raw, dim=1)
            att = att.view(batch_size, 1, att.shape[1])
            token_rep = torch.bmm(att, token_rep_raw)
        elif rep_type == "param":
            token_rep_raw = self._choose_span_context(embeddings, spans)
            att_param = torch.tanh(self.attention_map_repr[prot](self._choose_tokens(embeddings, tokens)))
            att_raw = torch.bmm(att_param.unsqueeze(1), token_rep_raw.view(batch_size, self.reduced_embedding_dim, token_rep_raw.shape[1]))
            att = F.softmax(att_raw, dim=1)
            token_rep = torch.bmm(att, token_rep_raw)

        return token_rep

    def _run_attention(self, prot, embeddings, tokens, spans, context_roots, context_spans):
        '''
            Various attention mechanisms implemented
        '''

        batch_size = embeddings.shape[0]

        tokens = torch.tensor(tokens, dtype=torch.long, device=self.device)
        # Get the required representation for pred/arg
        pure_token_rep = self._get_representation(prot=prot,
                                             embeddings=embeddings,
                                             tokens=tokens,
                                             spans=spans)
        # Concatenate root to representation
        if self.attention_type[prot]['repr'] != "root":
            token_rep = torch.cat((pure_token_rep, self._choose_tokens(batch=embeddings, lengths=tokens).unsqueeze(1)), dim=2)
        else:
            token_rep = pure_token_rep
        # Get the required representation for context of pred/arg
        context_type = self.attention_type[prot]['context']

        if context_type == "none":
            context_rep = None
        elif context_type == "param":
            sentence_rep = embeddings
            att_param = torch.tanh(self.attention_map_context[prot](pure_token_rep))
            att_raw = torch.bmm(att_param.unsqueeze(1), sentence_rep.view(batch_size, self.reduced_embedding_dim, sentence_rep.shape[1]))
            att = F.softmax(att_raw, dim=1)
            context_rep = torch.bmm(att, sentence_rep)
        elif context_type == "david":
            if prot == 'arg':
                prot_context = 'pred'
            else:
                prot_context = 'arg'

            if prot == "arg":
                context_rep = self._get_representation(
                    prot=prot_context, embeddings=embeddings,
                    tokens=context_roots, spans=context_spans)
            else:
                context_rep_args = []
                for ind in range(len(context_roots)):
                    c_roots = torch.tensor(context_roots[ind], dtype=torch.long, device=self.device).unsqueeze(0)
                    c_span = context_roots[ind]
                    embed_sent = embeddings[ind, :, :].unsqueeze(0)
                    c_rep_args = self._get_representation(
                        prot=prot_context, embeddings=embed_sent,
                        tokens=c_roots, spans=c_span)
                    context_rep_args = torch.stack(context_rep_args, dim=0)
                    att_context_param = torch.tanh(self.attention_map_context[prot](token_rep))
                    att_raw = torch.bmm(att_context_param.unsqueeze(1), context_rep_args.view(batch_size, self.reduced_embedding_dim, context_rep_args.shape[1]))
                    att = F.softmax(att_raw, dim=1)
                    context_rep = torch.bmm(att, token_rep)

        if context_rep is not None:
            inputs_for_regression = torch.cat((token_rep, context_rep.squeeze()), dim=1)
        else:
            inputs_for_regression = token_rep

        return inputs_for_regression

    def _run_regression(self, prot, h_in):
        h_out = {}
        for attr in self.all_attributes[prot]:
            h_out[attr] = h_in
            for i, lin_map in enumerate(self.lin_maps[prot][attr]):
                if i:
                    h_out[attr] = self._regression_nonlinearity(h_out[attr])
                h_out[attr] = lin_map(h_out[attr])
            h_out[attr] = h_out[attr].squeeze()
            h_out[attr] = torch.sigmoid(h_out[attr])
        return h_out

    def forward(self, prot, inputs, tokens, spans, context_roots,
                context_spans):
        """Forward propagation of activations"""

        inputs_for_attention = self._get_inputs(inputs)
        inputs_for_regression = self._run_attention(prot, inputs_for_attention,
                                                    tokens, spans,
                                                    context_roots,
                                                    context_spans)
        outputs = self._run_regression(prot, inputs_for_regression)

        return outputs


class MLPTrainer:

    loss_function_map = {"linear": MSELoss,
                         "robust": L1Loss,
                         "robust_smooth": SmoothL1Loss,
                         "multinomial": CrossEntropyLoss}

    def __init__(self, attention_type, regressiontype="multinomial",
                 optimizer_class=torch.optim.Adam, device="cpu", **kwargs):
        '''

        '''
        self._regressiontype = regressiontype
        self._optimizer_class = optimizer_class
        self._init_kwargs = kwargs
        self._continuous = regressiontype != "multinomial"
        self.device = device
        self.att_type = attention_type

    def _initialize_trainer_regression(self):
        '''

        '''
        lf_class = self.__class__.loss_function_map[self._regressiontype]
        if self._continuous:
            self._regression = MLPRegression(device=self.device,
                                             **self._init_kwargs)
            self._loss_function = lf_class()
        else:
            output_size = 2
            self._regression = MLPRegression(output_size=output_size,
                                             device=self.device,
                                             attention_type=self.att_type,
                                             **self._init_kwargs)
            self._loss_function = lf_class(reduction="none")

        self._regression = self._regression.to(self.device)
        self._loss_function = self._loss_function.to(self.device)

    def fit(self, X, Y, loss_wts, tokens, spans, context_roots, context_spans,
            dev, epochs):
        '''
            Fit X
        '''

        # Load the dev_data
        dev_x, dev_y, dev_tokens, dev_spans, dev_context_roots, dev_context_spans, dev_wts = [{}, {}, {}, {}, {}, {}, {}]
        for prot in ['arg', 'pred']:
            dev_x[prot], dev_y[prot], dev_tokens[prot], dev_spans[prot], dev_context_roots[prot], dev_context_spans[prot], dev_wts[prot] = dev[prot]

        self._initialize_trainer_regression()

        y_ = []
        loss_trace = []
        early_stop_acc = [0]

        parameters = [p for p in self._regression.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters)
        epoch = 0
        while epoch < epochs:
            epoch += 1
            print("Epoch", epoch, "of", epochs)
            for x, y, tks, sps, ctks, csps, wts in tqdm(zip(X, Y, tokens, spans, context_roots, context_spans, loss_wts), total=len(X)):
                if 'hyp' in list(y.keys()):
                    prot = "pred"
                else:
                    prot = "arg"
                attributes = list(y.keys())
                optimizer.zero_grad()
                losses = {}

                for attr in attributes:
                    if self._continuous:
                        y[attr] = torch.tensor(y[attr], dtype=torch.float, device=self.device)
                    else:
                        y[attr] = torch.tensor(y[attr], dtype=torch.long, device=self.device)
                    wts[attr] = torch.tensor(wts[attr], dtype=torch.float, device=self.device)

                y_ = self._regression(prot=prot, inputs=x, tokens=tks,
                                      spans=sps, context_roots=ctks,
                                      context_spans=csps)

                for attr in attributes:
                    losses[attr] = self._loss_function(y_[attr], y[attr])
                    if not self._continuous:
                        losses[attr] = torch.mm(losses[attr].unsqueeze(0), wts[attr].unsqueeze(1)).squeeze() / len(losses[attr])
                loss = sum(losses.values())
                loss.backward()
                optimizer.step()

                loss_trace.append(float(loss.data))

            # EARLY STOPPING
            # Perform validation run
            dev_preds = {}
            for prot in ['arg', 'pred']:
                dev_attributes = dev_y[prot].keys()
                dev_preds[prot] = self.predict(prot=prot,
                                               attributes=dev_attributes,
                                               X=dev_x[prot],
                                               tokens=dev_tokens[prot],
                                               spans=dev_spans[prot],
                                               context_roots=dev_context_roots[prot],
                                               context_spans=dev_context_spans[prot])
            print("Dev Metrics(Unweighted, Weighted)")
            early_stop_acc.append(self._print_metric(loss_trace=loss_trace,
                                                     dev_preds=dev_preds,
                                                     dev_y=dev_y,
                                                     dev_wts=dev_wts))
            y_ = []
            loss_trace = []
            if early_stop_acc[-1] - early_stop_acc[-2] < 0:
                break
            else:
                name_of_model = (self.att_type['arg']['repr'] + "_" +
                                 self.att_type['arg']['context'] + "_" +
                                 self.att_type['pred']['repr'] + "_" +
                                 self.att_type['pred']['context'] + "_" +
                                 str(epoch))
                Path = expanduser('~') + "/Desktop/saved_models/" + name_of_model
                torch.save(self._regression.state_dict(), Path)

    def _print_metric(self, loss_trace, dev_preds, dev_y, dev_wts):
        '''

        '''
        if self._continuous:
            sigdig = 3

            dev_r2 = {}
            dev_mse = {}
            for prot in ["arg", "pred"]:
                attributes = list(dev_y[prot][0].keys())
                for attr in attributes:
                    dev_r2[attr] = (np.round(r2(dev_y[prot][attr], dev_preds[prot][attr]), sigdig), np.round(r2(dev_y[prot][attr], dev_preds[prot][attr], sample_weight=dev_wts[attr]), sigdig))
                    dev_mse[attr] = (np.round(mse(dev_y[prot][attr], dev_preds[prot][attr]), sigdig), np.round(mse(dev_y[prot][attr], dev_preds[prot][attr], sample_weight=dev_wts[attr]), sigdig))

                print('Total loss:\t', np.round(np.mean(loss_trace), sigdig),
                      '\n', 'R2 DEV:\t', dev_r2, '\n',
                      'MSE DEV:\t', dev_mse, '\n')
        else:
            sigdig = 3
            print('Total loss:\t', np.round(np.mean(loss_trace), sigdig), '\n')
            for prot in ['arg', 'pred']:
                dev_f1 = {}
                dev_prec = {}
                dev_recall = {}
                dev_acc = {}
                mode_guess_dev = {}
                attributes = list(dev_y[prot].keys())
                for attr in attributes:
                    dev_f1[attr] = (np.round(f1(dev_y[prot][attr], dev_preds[prot][attr]), sigdig), np.round(f1(dev_y[prot][attr], dev_preds[prot][attr], sample_weight=dev_wts[prot][attr]), sigdig))
                    dev_prec[attr] = (np.round(prec(dev_y[prot][attr], dev_preds[prot][attr]), sigdig), np.round(prec(dev_y[prot][attr], dev_preds[prot][attr], sample_weight=dev_wts[prot][attr]), sigdig))
                    dev_recall[attr] = (np.round(rec(dev_y[prot][attr], dev_preds[prot][attr]), sigdig), np.round(rec(dev_y[prot][attr], dev_preds[prot][attr], sample_weight=dev_wts[prot][attr]), sigdig))
                    dev_acc[attr] = (np.round(acc(dev_y[prot][attr], dev_preds[prot][attr]), sigdig), np.round(acc(dev_y[prot][attr], dev_preds[prot][attr], sample_weight=dev_wts[prot][attr]), sigdig))
                    mode_dev = mode(dev_y[prot][attr])
                    mode_guess_dev[attr] = (np.round(acc(dev_y[prot][attr], [mode_dev[0] for i in range(len(dev_y[prot][attr]))]), sigdig), np.round(acc(dev_y[prot][attr], [mode_dev[0] for i in range(len(dev_y[prot][attr]))], sample_weight=dev_wts[prot][attr]), sigdig))

                print(prot, '\n',
                  'MODE ACC:\t', mode_guess_dev, '\n',
                  'ACCURACY:\t', dev_acc, sum([i for i, j in list(dev_acc.values())]), '\n',
                  'PRECISION:\t', dev_prec, '\n',
                  'RECALL:\t', dev_recall, '\n',
                  'F1 SCORE:\t', dev_f1, '\n')

            return sum([i for i, j in list(dev_acc.values())])

    def predict(self, prot, attributes, X, tokens, spans, context_roots, context_spans):
        """Predict using the MLP regression

        Parameters
        ----------
        X : iterable(iterable(object))
            a matrix of structures (independent variables) with rows
            corresponding to a particular kind of RNN
        """
        predictions = defaultdict(partial(np.ndarray, 0))
        for x, tokens_, spans_, ctx_root, ctx_span in zip(X, tokens, spans, context_roots, context_spans):

            tokens_ = torch.tensor(tokens_, dtype=torch.long, device=self.device)
            y_dev = self._regression(prot=prot, inputs=x, tokens=tokens_,
                                     spans=spans_, context_roots=ctx_root,
                                     context_spans=ctx_span)
            for attr in attributes:
                if self._continuous:
                    predictions[attr] = np.concatenate([predictions[attr], y_dev[attr].detach().cpu().numpy()])
                else:
                    predictions[attr] = np.concatenate([predictions[attr], torch.max(y_dev[attr], 1)[1].detach().cpu().numpy()])
            del y_dev
        return predictions
