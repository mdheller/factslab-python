from torch.nn import Module, Linear, ModuleDict, ModuleList
from torch.nn import MSELoss, L1Loss, SmoothL1Loss, CrossEntropyLoss, Dropout
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
                 embedding_dim=1024, output_size=3, layers=0):
        '''
            Super class for training
        '''
        super(MLPRegression, self).__init__()

        # Set model constants and embeddings
        self.device = device
        self.layers = layers
        self.embedding_dim = embedding_dim
        self.output_size = output_size
        self.attention_type = attention_type
        self.all_attributes = all_attrs

        # Initialise embeddings
        self._init_embeddings(embed_params)

        # Initialise regression layers and parameters
        self._init_regression()

        # Initialise attention parameters
        self._init_attention()

    def _init_embeddings(self, embedding_params):
        '''
            Initialise embeddings
        '''
        if type(embedding_params[0]) is str:
            self.vocab = None
            options_file = embedding_params[0]
            weight_file = embedding_params[1]
            self.embeddings = ElmoEmbedder(options_file, weight_file, cuda_device=0)
            self.reduced_embedding_dim = int(self.embedding_dim / 4)

        else:
            # GloVe embeddings
            glove_embeds = embedding_params[0]
            self.vocab = embedding_params[1]
            self.num_embeddings = len(self.vocab)
            self.embeddings = torch.nn.Embedding(self.num_embeddings,
                                                 self.embedding_dim,
                                                 max_norm=None,
                                                 norm_type=2,
                                                 scale_grad_by_freq=False,
                                                 sparse=False)
            self.reduced_embedding_dim = int(self.embedding_dim / 2)

            self.embeddings.weight.data.copy_(torch.from_numpy(glove_embeds.values))
            self.embeddings.weight.requires_grad = False
            self.vocab_hash = {w: i for i, w in enumerate(self.vocab)}

    def _init_regression(self):
        '''
            Define the linear maps
        '''
        if not self.vocab:
        # ELMO tuning parameters
            self.embed_linmap_argpred_lower = Linear(self.embedding_dim, self.reduced_embedding_dim)
            self.embed_linmap_argpred_mid = Linear(self.embedding_dim, self.reduced_embedding_dim, bias=False)
            self.embed_linmap_argpred_top = Linear(self.embedding_dim, self.reduced_embedding_dim, bias=False)
        else:
            self.embed_linmap = Linear(self.embedding_dim, self.reduced_embedding_dim)

        # Output regression parameters
        self.lin_maps = ModuleDict({'arg': ModuleList([]),
                                    'pred': ModuleList([])})
        self.layer_norm = ModuleDict({})

        for prot in self.all_attributes.keys():
            last_size = self.reduced_embedding_dim
            # Handle varying size of dimension depending on representation
            if self.attention_type[prot]['repr'] == "root":
                if self.attention_type[prot]['context'] != "none":
                    last_size *= 2
            else:
                if self.attention_type[prot]['context'] == "none":
                    last_size *= 1
                else:
                    last_size *= 3
            self.layer_norm[prot] = torch.nn.LayerNorm(last_size)
            for i in range(self.layers):
                out_size = int(last_size / ((i + 1) * 8))
                linmap = Linear(last_size, out_size)
                self.lin_maps[prot].append(linmap)
                last_size = out_size
            linmap = Linear(last_size, self.output_size)
            self.lin_maps[prot].append(linmap)

        # Dropout layer
        self.dropout = Dropout(p=0.2)

    def _regression_nonlinearity(self, x):
        return F.relu(x)

    def _init_attention(self):
        '''
            Initialises the attention map vector/matrix

            Takes attention_type-Span, Sentence, Span-param, Sentence-param
            as a parameter to decide the size of the attention matrix
        '''

        self.att_map_repr = ModuleDict({})
        self.att_map_v = ModuleDict({})
        self.att_map_context = ModuleDict({})
        for prot in self.attention_type.keys():
            # Token representation
            if self.attention_type[prot]['repr'] == "span":
                self.att_map_repr[prot] = Linear(self.reduced_embedding_dim, 1, bias=False)
            elif self.attention_type[prot]['repr'] == "param":
                self.att_map_repr[prot] = Linear(self.reduced_embedding_dim, self.reduced_embedding_dim, bias=False)

            # Context representation
            if self.attention_type[prot]['repr'] in ["span", "param"]:
                repr_dim = 2 * self.reduced_embedding_dim
            else:
                repr_dim = self.reduced_embedding_dim
            if self.attention_type[prot]['context'] == "david":
                if prot == "pred":
                    self.att_map_context[prot] = Linear(repr_dim, self.reduced_embedding_dim, bias=False)
            elif self.attention_type[prot]['context'] == "param":
                self.att_map_context[prot] = Linear(repr_dim, self.reduced_embedding_dim, bias=False)

    def _choose_tokens(self, batch, lengths):
        '''
            Extracts tokens from a batch at specified position(lengths)
            batch - batch_size x max_sent_length x embed_dim
            lengths - batch_size x max_span_length x embed_dim
        '''
        idx = (lengths).unsqueeze(2).expand(-1, -1, batch.shape[2])
        return batch.gather(1, idx).squeeze()

    def _get_inputs(self, words):
        '''Return ELMO embeddings
            Can be done either as a module, or programmatically

            If done programmatically, the 3 layer representations are concatenated, then mapped to a lower dimension and squashed with tanh
        '''
        if not self.vocab:
            raw_embeds, masks = self.embeddings.batch_to_embeddings(words)
            masks = masks.unsqueeze(2).repeat(1, 1, self.reduced_embedding_dim).byte()
            embedded_inputs = (
                self.embed_linmap_argpred_lower(raw_embeds[:, 0, :, :].squeeze()) +
                self.embed_linmap_argpred_mid(raw_embeds[:, 1, :, :].squeeze()) +
                self.embed_linmap_argpred_top(raw_embeds[:, 2, :, :].squeeze()))
            masked_embedded_inputs = embedded_inputs * masks.float()
            return masked_embedded_inputs, masks
        else:
            # Glove embeddings
            indices = [[self.vocab_hash[word] for word in sent] for sent in words]
            indices = torch.tensor(indices, dtype=torch.long, device=self.device)
            embeddings = self.embeddings(indices)
            masks = (embeddings != 0)[:, :, :self.reduced_embedding_dim].byte()
            reduced_embeddings = self.embed_linmap(embeddings) * masks.float()
            return reduced_embeddings, masks

    def _get_representation(self, prot, embeddings, tokens, spans, masks):
        '''
            returns the representation required from arguments passed by
            running attention based on arguments passed
        '''

        # Get token(pred/arg) representation
        rep_type = self.attention_type[prot]['repr']

        if rep_type == "root":
            token_rep = self._choose_tokens(batch=embeddings, lengths=tokens)
            if len(token_rep.shape) == 1:
                token_rep = token_rep.unsqueeze(0)
        else:
            masks_spans = (spans == -1)
            spans[spans == -1] = 0
            roots_rep_raw = self._choose_tokens(embeddings, tokens)
            spans_rep_raw = self._choose_tokens(embeddings, spans)
            if len(roots_rep_raw.shape) == 1:
                roots_rep_raw = roots_rep_raw.unsqueeze(0)
            if rep_type == "span":
                att_raw = self.att_map_repr[prot](spans_rep_raw).squeeze()
            elif rep_type == "param":
                att_param = torch.tanh(self.att_map_repr[prot](roots_rep_raw)).unsqueeze(2)
                att_raw = torch.matmul(spans_rep_raw, att_param).squeeze()

            att_raw = att_raw.masked_fill(masks_spans, -1e9)
            att = F.softmax(att_raw, dim=1)
            # att = self.dropout(att)
            token_rep = torch.matmul(att.unsqueeze(2).permute(0, 2, 1),
                                     spans_rep_raw).squeeze()

        return token_rep

    def _run_attention(self, prot, embeddings, tokens, spans, context_roots, context_spans, masks):
        '''
            Various attention mechanisms implemented
        '''

        # Get the required representation for pred/arg
        pure_token_rep = self._get_representation(prot=prot,
                                             embeddings=embeddings,
                                             tokens=tokens,
                                             spans=spans,
                                             masks=masks)
        # Concatenate root to representation
        # if self.attention_type[prot]['repr'] != "root":
        #     root_embedding = self._choose_tokens(batch=embeddings, lengths=tokens)
        #     token_rep = torch.stack((root_embedding, pure_token_rep), dim=2).view(embeddings.shape[0], self.reduced_embedding_dim * 2)
        # else:
        token_rep = pure_token_rep

        # Get the required representation for context of pred/arg
        context_type = self.attention_type[prot]['context']

        if context_type == "none":
            context_rep = None
        elif context_type == "param":
            # Sentence level attention
            att_param = torch.tanh(self.att_map_context[prot](token_rep)).unsqueeze(1)
            att_raw = torch.matmul(embeddings, att_param.permute(0, 2, 1))
            att_raw = att_raw.masked_fill(masks[:, :, 0:1] == 0, -1e9)
            att = F.softmax(att_raw, dim=1)
            # att = self.dropout(att)
            context_rep = torch.matmul(att.permute(0, 2, 1), embeddings)
        elif context_type == "david":
            if prot == 'arg':
                prot_context = 'pred'
            else:
                prot_context = 'arg'

            if prot == "arg":
                context_roots = torch.tensor(context_roots, dtype=torch.long, device=self.device)
                context_rep = self._get_representation(
                    prot=prot_context, embeddings=embeddings,
                    tokens=context_roots, spans=context_spans, masks=masks)
            else:
                context_rep = None
                for i, context_root in enumerate(context_roots):
                    ctx_reps = None
                    context_root = torch.tensor(context_root, dtype=torch.long, device=self.device).unsqueeze(0)
                    context_span = context_spans[i]
                    sentence = embeddings[i, :, :].unsqueeze(0)
                    for j in range(len(context_span)):
                        ctx_root = context_root[:, j].unsqueeze(1)
                        ctx_span = [context_span[j]]
                        ctx_rep = self._get_representation(
                            prot=prot_context, embeddings=sentence,
                            tokens=ctx_root, spans=ctx_span, masks=masks)

                        if j:
                            ctx_reps = torch.cat((ctx_reps, ctx_rep), dim=0)
                        else:
                            ctx_reps = ctx_rep

                    # Attention over arguments
                    att_nd_param = torch.tanh(self.att_map_context[prot](token_rep[i, :].unsqueeze(0)))
                    att_raw = torch.mm(att_nd_param, ctx_reps.view(ctx_reps.shape[1], ctx_reps.shape[0]))
                    att = F.softmax(att_raw, dim=1)
                    ctx_rep_final = torch.mm(att, ctx_reps)
                    if i:
                        context_rep = torch.cat((context_rep, ctx_rep_final), dim=0).squeeze()
                    else:
                        context_rep = ctx_rep_final

        if context_rep is not None:
            inputs_for_regression = torch.cat((token_rep.squeeze(), context_rep.squeeze()), dim=1)
        else:
            inputs_for_regression = token_rep

        return inputs_for_regression

    def _run_regression(self, prot, h_in):
        '''
            Run regression to get 3 attribute vector
        '''
        h_out = h_in
        for i, lin_map in enumerate(self.lin_maps[prot]):
            if i:
                h_out = self._regression_nonlinearity(h_out)
                h_out = self.dropout(h_out)
            h_out = lin_map(h_out)

        # h_out = torch.sigmoid(h_out).squeeze()
        # Now add another dimension with 1 - probability for False
        # h_out = h_out.unsqueeze(2).repeat(1, 1, 2)
        # h_out[:, :, 1] = 1 - h_out[:, :, 1]
        h_out = h_out.view(-1, 3, 2)
        final_h = {}
        for i, attr in enumerate(self.all_attributes[prot]):
            final_h[attr] = h_out[:, i, :]
        return final_h

    def forward(self, prot, words, tokens, spans, context_roots,
                context_spans):
        """Forward propagation of activations"""
        inputs_for_attention, masks = self._get_inputs(words)
        inputs_for_regression = self._run_attention(prot=prot,
                                    embeddings=inputs_for_attention,
                                    tokens=tokens, spans=spans,
                                    context_roots=context_roots,
                                    context_spans=context_spans,
                                    masks=masks)
        outputs = self._run_regression(prot=prot, h_in=inputs_for_regression)

        return outputs


class MLPTrainer:

    loss_function_map = {"linear": MSELoss,
                         "robust": L1Loss,
                         "robust_smooth": SmoothL1Loss,
                         "multinomial": CrossEntropyLoss}

    def __init__(self, attention_type, all_attrs, regressiontype="multinomial",
                 optimizer_class=torch.optim.Adam, device="cpu",
                 lr=0.001, weight_decay=0, **kwargs):
        '''

        '''
        self._regressiontype = regressiontype
        self._optimizer_class = optimizer_class
        self._init_kwargs = kwargs
        self._continuous = regressiontype != "multinomial"
        self.device = device
        self.att_type = attention_type
        self.all_attrs = all_attrs
        self.lr = lr
        self.weight_decay = weight_decay

    def _initialize_trainer_regression(self):
        '''

        '''
        lf_class = self.__class__.loss_function_map[self._regressiontype]
        if self._continuous:
            self._regression = MLPRegression(device=self.device,
                                             **self._init_kwargs)
            self._loss_function = lf_class()
        else:
            output_size = 6
            self._regression = MLPRegression(output_size=output_size,
                                             device=self.device,
                                             attention_type=self.att_type,
                                             all_attrs=self.all_attrs,
                                             **self._init_kwargs)
            self._loss_function = lf_class(reduction="none")

        self._regression = self._regression.to(self.device)
        # self._regression = torch.nn.DataParallel(self._regression, device_ids=[0, 1, 2], output_device=0)
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
        optimizer = torch.optim.Adam(parameters, weight_decay=self.weight_decay, lr=self.lr)
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

                tks = torch.tensor(tks, dtype=torch.long, device=self.device).unsqueeze(1)
                max_span = max([len(a) for a in sps])
                sps = torch.tensor([a + [-1 for i in range(max_span - len(a))] for a in sps], dtype=torch.long, device=self.device)

                for attr in attributes:
                    if self._continuous:
                        y[attr] = torch.tensor(y[attr], dtype=torch.float, device=self.device)
                    else:
                        y[attr] = torch.tensor(y[attr], dtype=torch.long, device=self.device)
                    wts[attr] = torch.tensor(wts[attr], dtype=torch.float, device=self.device)

                y_ = self._regression(prot=prot, words=x, tokens=tks,
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
            dev_preds = {}
            self._regression = self._regression.eval()
            for mj, prot in enumerate(['arg', 'pred']):
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
            self._regression = self._regression.train()
            # if early_stop_acc[-1] - early_stop_acc[-2] < 0:
            #     break
            # else:
            name_of_model = (str(self._regression.embedding_dim) +
                             self.att_type['arg']['repr'] + "_" +
                             self.att_type['arg']['context'] + "_" +
                             self.att_type['pred']['repr'] + "_" +
                             self.att_type['pred']['context'] + "_" +
                             str(epoch))
            Path = expanduser('~') + "/Desktop/saved_models/" + name_of_model
            torch.save(self._regression.state_dict(), Path)

    def _print_metric(self, loss_trace, dev_preds, dev_y, dev_wts):
        '''

        '''
        return_val = 0
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
                mode_dev = {}
                mode_guess_dev = {}
                attributes = list(dev_y[prot].keys())
                for attr in attributes:
                    mode_dev[attr] = mode(dev_y[prot][attr])[0][0]
                    dev_f1[attr] = (np.round(f1(dev_y[prot][attr], dev_preds[prot][attr], pos_label=mode_dev[attr]), sigdig), np.round(f1(dev_y[prot][attr], dev_preds[prot][attr], sample_weight=dev_wts[prot][attr], pos_label=mode_dev[attr]), sigdig))
                    dev_prec[attr] = (np.round(prec(dev_y[prot][attr], dev_preds[prot][attr], pos_label=mode_dev[attr]), sigdig), np.round(prec(dev_y[prot][attr], dev_preds[prot][attr], sample_weight=dev_wts[prot][attr], pos_label=mode_dev[attr]), sigdig))
                    dev_recall[attr] = (np.round(rec(dev_y[prot][attr], dev_preds[prot][attr], pos_label=mode_dev[attr]), sigdig), np.round(rec(dev_y[prot][attr], dev_preds[prot][attr], sample_weight=dev_wts[prot][attr], pos_label=mode_dev[attr]), sigdig))
                    dev_acc[attr] = (np.round(acc(dev_y[prot][attr], dev_preds[prot][attr]), sigdig), np.round(acc(dev_y[prot][attr], dev_preds[prot][attr], sample_weight=dev_wts[prot][attr]), sigdig))

                    mode_guess_dev[attr] = (np.round(acc(dev_y[prot][attr], [mode_dev[attr] for i in range(len(dev_y[prot][attr]))]), sigdig), np.round(acc(dev_y[prot][attr], [mode_dev[attr] for i in range(len(dev_y[prot][attr]))], sample_weight=dev_wts[prot][attr]), sigdig))

                print(prot, "\n",
                  mode_dev, "\n",
                  "ACC MODE :\t", mode_guess_dev, "\n",
                  "ACCURACY:\t", dev_acc, sum([i for i, j in list(dev_acc.values())]), "\n",
                  "PRECISION:\t", dev_prec, "\n",
                  "RECALL:\t", dev_recall, "\n",
                  "F1 SCORE:\t", dev_f1, "\n\n")

                return_val += sum([i for i, j in list(dev_acc.values())])

            return return_val

    def predict(self, prot, attributes, X, tokens, spans, context_roots, context_spans):
        """Predict using the MLP regression

        Parameters
        ----------
        X : iterable(iterable(object))
            a matrix of structures (independent variables) with rows
            corresponding to a particular kind of RNN
        """
        self._regression = self._regression.eval()
        predictions = defaultdict(partial(np.ndarray, 0))
        for x, tks, sps, ctx_root, ctx_span in zip(X, tokens, spans, context_roots, context_spans):

            tks = torch.tensor(tks, dtype=torch.long, device=self.device).unsqueeze(1)
            max_span = max([len(a) for a in sps])
            sps = torch.tensor([a + [-1 for i in range(max_span - len(a))] for a in sps], dtype=torch.long, device=self.device)

            y_dev = self._regression(prot=prot, words=x, tokens=tks,
                                     spans=sps, context_roots=ctx_root,
                                     context_spans=ctx_span)
            for attr in attributes:
                if self._continuous:
                    predictions[attr] = np.concatenate([predictions[attr], y_dev[attr].detach().cpu().numpy()])
                else:
                    predictions[attr] = np.concatenate([predictions[attr], torch.max(y_dev[attr], 1)[1].detach().cpu().numpy()])
            del y_dev
        return predictions
