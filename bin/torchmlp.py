import torch
from torch.nn import Module, Linear, ModuleList, Dropout, L1Loss, BCELoss
import numpy as np
import pickle
import argparse
import random
import sys
from sklearn.metrics import accuracy_score as accuracy, mean_absolute_error as mae
from itertools import product
from ast import literal_eval
from factslab.utility import print_metrics as print_metrics

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)


class simpleMLP(Module):
    def __init__(self, device, input_size=300, layers=(512, 256),
                 output_size=3, p_dropout=0.5, activation='relu',
                 continuous=True):
        super(simpleMLP, self).__init__()
        self._device = device
        self._linmaps = ModuleList([])
        last_size = input_size
        for j in layers:
            self._linmaps.append(Linear(last_size, j))
            last_size = j
        self._linmaps.append(Linear(last_size, output_size))
        self._dropout = Dropout(p=p_dropout)
        self._activation = activation
        self._continuous = True

    def nonlinearity(self, x):
        if self._activation == 'relu':
            return torch.nn.functional.relu(x)
        elif self._activation == 'tanh':
            return torch.tanh(x)

    def forward(self, x, return_hidden=False):
        '''
            Runs forward pass on neural network

            Parameters:
            ----------
            x: input to neural network
            return_hidden: if true return a list of hidden state activations

            Returns:
            ------
            torch.sigmoid(x): Squashed final layer activation
            hidden: Hidden layer activations(if return_hidden is True)
        '''
        hidden = []
        for i, linmap in enumerate(self._linmaps):
            if i:
                x = self.nonlinearity(x)
                x = self._dropout(x)
            x = linmap(x)
            hidden.append(x.detach().cpu().numpy())
        if return_hidden:
            if not self._continuous:
                return torch.sigmoid(x), hidden
            else:
                return torch.sigmoid(x), hidden
                # return x, hidden
        else:
            if not self._continuous:
                return torch.sigmoid(x)
            else:
                # return x
                return torch.sigmoid(x)


def main(prot, batch_size, elmo_on, glove_on, typeabl, tokenabl, type_on,
         token_on, search_on, test_on, best_params, weighted, regression_type):

    if args.prot == "arg":
        attributes = ["part", "kind", "abs"]
        attr_map = {"part": "Is.Particular", "kind": "Is.Kind", "abs": "Is.Abstract"}
        attr_conf = {"part": "Part.Confidence", "kind": "Kind.Confidence",
                 "abs": "Abs.Confidence"}
    else:
        attributes = ["part", "hyp", "dyn"]
        attr_map = {"part": "Is.Particular", "dyn": "Is.Dynamic", "hyp": "Is.Hypothetical"}
        attr_conf = {"part": "Part.Confidence", "dyn": "Dyn.Confidence",
                 "hyp": "Hyp.Confidence"}

    path = '/data/venkat/pickled_data_' + regression_type + '/' + prot

    with open(path + 'hand.pkl', 'rb') as fin:
        x_pd, y, dev_x_pd, dev_y, test_x_pd, test_y, data, data_dev_mean, data_test_mean, feature_names = pickle.load(fin)
    verbnet_classids, supersenses, frame_names, lcs_feats, conc_cols, lexical_feats, all_ud_feature_cols = feature_names
    type_cols = verbnet_classids + supersenses + frame_names + lcs_feats + conc_cols
    token_cols = lexical_feats + all_ud_feature_cols
    type_abl_dict = {0: [], 1: verbnet_classids, 2: supersenses, 3: frame_names, 4: lcs_feats, 5: conc_cols}
    type_abl_names = {0: [], 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
    token_abl_dict = {0: 'None', 1: ['UPOS='], 2: ['XPOS='], 3: ['UPOS=', 'XPOS=', 'DEPRE'], 4: ['DEPRE'], 5: ['UPOS=', 'XPOS='], 6: 'Lexical'}

    if type_on or token_on:
        ablation = []
        if type_on and not token_on:
            type_feats_to_remove = type_abl_dict[typeabl]
            ablation = token_cols + type_feats_to_remove
        elif token_on and not type_on:
            if tokenabl == 6:
                token_feats_to_remove = lexical_feats
            elif tokenabl == 3:
                token_feats_to_remove = [a for a in all_ud_feature_cols if a[0:5] not in token_abl_dict[tokenabl]]
            else:
                token_feats_to_remove = [a for a in all_ud_feature_cols if a[0:5] in token_abl_dict[tokenabl]]
            ablation = type_cols + token_feats_to_remove

        cols_to_drop = x_pd.columns[(x_pd == 0).all()].values.tolist()
        ablation = ablation + cols_to_drop
        x_hand = x_pd.drop(x_pd.columns.intersection(ablation), axis=1).values
        dev_x_hand = dev_x_pd.drop(dev_x_pd.columns.intersection(ablation), axis=1).values
        test_x_hand = test_x_pd.drop(test_x_pd.columns.intersection(ablation), axis=1).values

    else:
        x_hand = None
        dev_x_hand = None
        test_x_hand = None

    if elmo_on:
        with open(path + 'train_elmo.pkl', 'rb') as tr_elmo, open(path + 'dev_elmo.pkl', 'rb') as dev_elmo, open(path + 'test_elmo.pkl', 'rb') as test_elmo:
            x_elmo = pickle.load(tr_elmo)
            dev_x_elmo = pickle.load(dev_elmo)
            test_x_elmo = pickle.load(test_elmo)
    else:
        x_elmo = None
        dev_x_elmo = None
        test_x_elmo = None
    if glove_on:
        with open(path + 'train_glove.pkl', 'rb') as tr_glove, open(path + 'dev_glove.pkl', 'rb') as dev_glove, open(path + 'test_glove.pkl', 'rb') as test_glove:
            x_glove = pickle.load(tr_glove)
            dev_x_glove = pickle.load(dev_glove)
            test_x_glove = pickle.load(test_glove)
    else:
        x_glove = None
        dev_x_glove = None
        test_x_glove = None

    try:
        x = np.concatenate([a for a in (x_hand, x_elmo, x_glove) if a is not None], axis=1)
        dev_x = np.concatenate([a for a in (dev_x_hand, dev_x_elmo, dev_x_glove) if a is not None], axis=1)
        test_x = np.concatenate([a for a in (test_x_hand, test_x_elmo, test_x_glove) if a is not None], axis=1)
    except ValueError:
        sys.exit('You need an input representation')

    # import matplotlib
    # matplotlib.use('agg')
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # fig, ax = plt.subplots(figsize=[20, 5], nrows=1, ncols=3, squeeze=False, sharey='row')
    # plt.suptitle(prot)
    # for i in range(3):
    #     ax[0][i].set_title(attributes[i])
    #     sns.distplot(y[attributes[i]], ax=ax[0][i]).get_figure()
    # plt.savefig(prot + '.png')
    # sys.exit(0)

    device = torch.device('cuda:0')
    grid_params = {'hidden_layer_sizes': [(512, 256), (512, 128), (512, 64), (512, 32), (512,), (256, 128), (256, 64), (256, 32), (256,), (128, 64), (128, 32), (128,), (64, 32), (64,), (32,)], 'alpha': [0, 0.00001, 0.0001, 0.001], 'dropout': [0.1, 0.2, 0.3, 0.4, 0.5], 'activation': ['relu']}
    all_grid_params = list(product(grid_params['hidden_layer_sizes'], grid_params['alpha'], grid_params['dropout'], grid_params['activation']))

    # Minibatch y, x and dev_x
    y = [[y[attributes[0]][i], y[attributes[1]][i], y[attributes[2]][i]] for i in range(len(y[attributes[0]]))]
    y = [np.array(y[i:i + batch_size]) for i in range(0, len(y), batch_size)]
    dev_y = np.array([[dev_y[attributes[0]][i], dev_y[attributes[1]][i], dev_y[attributes[2]][i]] for i in range(len(dev_y[attributes[0]]))])
    test_y = np.array([[test_y[attributes[0]][i], test_y[attributes[1]][i], test_y[attributes[2]][i]] for i in range(len(test_y[attributes[0]]))])

    x = [np.array(x[i:i + batch_size]) for i in range(0, len(x), batch_size)]
    dev_x = [np.array(dev_x[i:i + batch_size]) for i in range(0, len(dev_x), batch_size)]
    test_x = [np.array(test_x[i:i + batch_size]) for i in range(0, len(test_x), batch_size)]

    loss_wts = data.loc[:, [(attr_conf[attr] + ".Norm") for attr in attributes]].values
    loss_wts = [np.array(loss_wts[i:i + batch_size]) for i in range(0, len(loss_wts), batch_size)]
    dev_loss_wts = data_dev_mean.loc[:, [(attr_conf[attr] + ".Norm") for attr in attributes]].values
    test_loss_wts = data_test_mean.loc[:, [(attr_conf[attr] + ".Norm") for attr in attributes]].values
    is_data_continuous = regression_type != "multinomial"

    if search_on:
        search_scores = []
        for (hidden_state, alpha, drp, act) in all_grid_params:
            clf = simpleMLP(device=device, input_size=x[0].shape[1],
                    layers=hidden_state,
                    output_size=y[0].shape[1], p_dropout=drp,
                    activation=act, continuous=is_data_continuous)
            clf.to(device)
            loss_function = BCELoss()
            parameters = [p for p in clf.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(parameters, weight_decay=alpha)
            early_stopping = [1000]
            for epoch in range(20):
                for x_, dev_x_, y_, wts in zip(x, dev_x, y, loss_wts):
                    optimizer.zero_grad()

                    x_ = torch.tensor(x_, dtype=torch.float, device=device)
                    dev_x_ = torch.tensor(dev_x_, dtype=torch.float, device=device)
                    y_ = torch.tensor(y_, dtype=torch.float, device=device)
                    wts = torch.tensor(wts, dtype=torch.float, device=device)

                    y_pred = clf(x_)
                    loss = loss_function(y_pred, y_)
                    # loss = torch.sum(loss * wts) / batch_size
                    loss.backward()
                    optimizer.step()
                    # loss_trace.append(float(loss.data))
                clf = clf.eval()
                y_pred_dev, _ = predict(clf, dev_x, device)
                clf = clf.train()
                if is_data_continuous:
                    score = mae(dev_y, y_pred_dev)
                else:
                    score = accuracy(dev_y, y_pred_dev)
                early_stopping.append(score)
                if early_stopping[-1] - early_stopping[-2] > 0:
                    search_scores.append((hidden_state, alpha, drp, act, early_stopping[-2]))
                    break
                else:
                    name_of_model = (prot + "_" +
                             "elmo:" + str(elmo_on) + "_" +
                             "glove:" + str(glove_on) + "_" +
                             "token:" + str(token_on) + "_" +
                             "type:" + str(type_on) + "_" +
                             "tokenabl:" + str(token_abl_dict[tokenabl]) +
                             "_" + "typeabl:" + str(type_abl_names[typeabl]) +
                             "_" + str(hidden_state) + "_" +
                             str(alpha) + "_" +
                             str(drp) + "_" + str(act))
                    Path = "/data/venkat/saved_models/" + regression_type + "/" + name_of_model
                    torch.save(clf.state_dict(), Path)
        print(min(search_scores, key=lambda x: x[-1]), "\n")
    else:
        best_params = literal_eval(best_params)
        hidden_state, alpha, drp, act, _ = best_params
        clf = simpleMLP(device=device, input_size=dev_x[0].shape[1],
                        layers=hidden_state, p_dropout=drp,
                        activation=act)
        clf.to(device)
        name_of_model = (prot + "_" +
                         "elmo:" + str(elmo_on) + "_" +
                         "glove:" + str(glove_on) + "_" +
                         "token:" + str(token_on) + "_" +
                         "type:" + str(type_on) + "_" +
                         "tokenabl:" + str(token_abl_dict[tokenabl]) +
                         "_" + "typeabl:" + str(type_abl_names[typeabl]) +
                         "_" + str(hidden_state) + "_" +
                         str(alpha) + "_" +
                         str(drp) + "_" + str(act))
        onoff_map = {True: '+', False: '-'}
        abl_state = "& " + " & ".join([onoff_map[xy] for xy in [type_on, token_on, glove_on, elmo_on]])
        best_model = "/data/venkat/saved_models/" + regression_type + "/" + name_of_model
        clf.load_state_dict(torch.load(best_model))
        clf.eval()
        if not test_on:
            y_pred_dev, h = predict(clf, dev_x, device)
            # y = np.concatenate(y, axis=0)
            print_metrics(attributes=attributes, attr_map=attr_map,
                          attr_conf=attr_conf, wts=dev_loss_wts,
                          y_true=y, y_pred=y_pred_dev, fstr=abl_state,
                          weighted=weighted, regression_type=regression_type)
            # do_riemann(h, dev_y)
        else:
            y_pred_test, _ = predict(clf, test_x, device)
            print_metrics(attributes=attributes, attr_map=attr_map,
                          attr_conf=attr_conf, wts=test_loss_wts,
                          y_true=test_y, y_pred=y_pred_test, fstr=abl_state,
                          weighted=weighted, regression_type=regression_type)


def predict(clf, x, device):
    predictions = np.empty((0, 3), int)
    final_h_size = clf._linmaps[-2].weight.shape[0]
    h = np.empty((0, final_h_size), float)
    for mb in x:
        mb = torch.tensor(mb, dtype=torch.float, device=device)
        preds, h_ = clf(mb, return_hidden=True)
        if not clf._continuous:
            preds = preds > 0.5
        predictions = np.concatenate([predictions, preds.detach().cpu().numpy()])
        h = np.concatenate([h, h_[-2]])
    return predictions, h


def do_riemann(h, y):
    import umap
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    reducer = umap.UMAP(random_state=1)
    reducer.fit(h)
    embedding = reducer.transform(h)
    # target_dict = {'[0 0 0]': 0, '[1 0 0]': 1, '[0 1 0]': 2, '[0 0 1]': 3, '[1 1 0]': 4, '[0 1 1]': 5, '[1 0 1]': 6, '[1 1 1]': 7}
    # targets = np.array([i for i in range(0, 8)])
    plt.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap='Spectral')
    # plt.gca().set_aspect('equal', 'datalim')
    # plt.legend()
    # plt.colorbar(boundaries=np.arange(8) - 0.5).set_ticks(np.arange(8))
    plt.savefig('umap.png')


if __name__ == '__main__':
    # initialize argument parser
    description = 'Run an RNN regression on Genericity protocol annotation.'
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--prot',
                        type=str,
                        default='arg')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32)
    parser.add_argument('--typeabl',
                        type=int,
                        default=0)
    parser.add_argument('--tokenabl',
                        type=int,
                        default=0)
    parser.add_argument('--elmo',
                        action='store_true',
                        help='Turn on elmo embeddings')
    parser.add_argument('--glove',
                        action='store_true',
                        help='Turn on glove embeddings')
    parser.add_argument('--token',
                        action='store_true',
                        help='Turn on token level features')
    parser.add_argument('--type',
                        action='store_true',
                        help='Turn on type level features')
    parser.add_argument('--search',
                        action='store_true',
                        help='Run grid search')
    parser.add_argument('--best',
                        type=str,
                        default=None)
    parser.add_argument('--test',
                        action='store_true',
                        help='Run test')
    parser.add_argument('--weighted',
                        action='store_true',
                        help='Run test')
    parser.add_argument('--regressiontype',
                        type=str,
                        default="regression",
                        help='regression or classification')

    args = parser.parse_args()

    main(prot=args.prot, batch_size=args.batch_size, glove_on=args.glove,
         elmo_on=args.elmo, token_on=args.token, type_on=args.type,
         tokenabl=args.tokenabl, typeabl=args.typeabl,
         search_on=args.search, test_on=args.test, best_params=args.best,
         weighted=args.weighted, regression_type=args.regressiontype)
