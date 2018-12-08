import torch
from torch.nn import Module, Linear, ModuleList, BCELoss, Dropout
import numpy as np
import pickle
import argparse
import random
from os.path import expanduser
import sys
from sklearn.metrics import accuracy_score as accuracy, precision_score as precision, recall_score as recall, f1_score as f1
from itertools import product

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)


class simpleMLP(Module):
    def __init__(self, device, input_size=300, layers=(512, 256),
                 output_size=3, p_dropout=0.5, activation='relu'):
        super(simpleMLP, self).__init__()
        self.device = device
        self.linmaps = ModuleList([])
        last_size = input_size
        for j in layers:
            self.linmaps.append(Linear(last_size, j))
            last_size = j
        self.linmaps.append(Linear(last_size, output_size))
        self.dropout = Dropout(p=p_dropout)
        self.activation = activation

    def nonlinearity(self, x):
        if self.activation == 'relu':
            return torch.nn.functional.relu(x)
        elif self.activation == 'tanh':
            return torch.tanh(x)

    def forward(self, x):
        for i, linmap in enumerate(self.linmaps):
            if i:
                x = self.nonlinearity(x)
                x = self.dropout(x)
            x = linmap(x)
        return torch.sigmoid(x)


def main(prot, batch_size, elmo_on, glove_on, abl, tokenabl, type_on, token_on, search_on, test_on):

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

    token_abl_names = {0: 'None', 1: ['UPOS='], 2: ['XPOS='], 3: ['UPOS=', 'XPOS=', 'DEPRE'], 4: ['DEPRE'], 5: ['UPOS=', 'XPOS=']}

    with open('data/' + prot + 'hand.pkl', 'rb') as fin:
        x_pd, y, dev_x_pd, dev_y, test_x_pd, test_y, data, data_dev_mean, data_test_mean, feature_names = pickle.load(fin)
    verbnet_classids, supersenses, frame_names, lcs_feats, conc_cols, lexical_feats, all_ud_feature_cols = feature_names
    type_cols = verbnet_classids + supersenses + frame_names + lcs_feats + conc_cols
    token_cols = lexical_feats + all_ud_feature_cols
    abl_dict = {0: [], 1: verbnet_classids, 2: supersenses, 3: frame_names, 4: lcs_feats, 5: conc_cols, 6: lexical_feats, 7: all_ud_feature_cols}

    if type_on or token_on:
        if type_on and not token_on and not abl and not tokenabl:
            ablation = token_cols
        elif token_on and not type_on and not abl:
            if not tokenabl:
                ablation = type_cols
            else:
                if tokenabl != 3:
                    ud_feats_to_remove = [a for a in all_ud_feature_cols if a[0:5] in token_abl_names[tokenabl]]
                else:
                    ud_feats_to_remove = [a for a in all_ud_feature_cols if a[0:5] not in token_abl_names[tokenabl]]
                ablation = type_cols + lexical_feats + ud_feats_to_remove
        else:
            ablation = abl_dict[abl]
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
        with open('data/' + prot + 'train_elmo.pkl', 'rb') as tr_elmo, open('data/' + prot + 'dev_elmo.pkl', 'rb') as dev_elmo, open('data/' + prot + 'test_elmo.pkl', 'rb') as test_elmo:
            x_elmo = pickle.load(tr_elmo)
            dev_x_elmo = pickle.load(dev_elmo)
            test_x_elmo = pickle.load(test_elmo)
    else:
        x_elmo = None
        dev_x_elmo = None
        test_x_elmo = None
    if glove_on:
        with open('data/' + prot + 'train_glove.pkl', 'rb') as tr_glove, open('data/' + prot + 'dev_glove.pkl', 'rb') as dev_glove, open('data/' + prot + 'test_glove.pkl', 'rb') as test_glove:
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

    device = torch.device('cuda:0')
    grid_params = {'hidden_layer_sizes': [(512, 256), (512, 128), (512, 64), (512, 32), (512,), (256, 128), (256, 64), (256, 32), (256,), (128, 64), (128, 32), (128,), (64, 32), (64,), (32,)], 'alpha': [0, 0.00001, 0.0001, 0.001], 'dropout': [0.1, 0.2, 0.3, 0.4, 0.5], 'activation': ['tanh', 'relu']}
    all_grid_params = list(product(grid_params['hidden_layer_sizes'], grid_params['alpha'], grid_params['dropout'], grid_params['activation']))

    # Minibatch y, x and dev_x
    y = [[y[attributes[0]][i], y[attributes[1]][i], y[attributes[2]][i]] for i in range(len(y[attributes[0]]))]
    y = [np.array(y[i:i + batch_size]) for i in range(0, len(y), batch_size)]
    dev_y = np.array([[dev_y[attributes[0]][i], dev_y[attributes[1]][i], dev_y[attributes[2]][i]] for i in range(len(dev_y[attributes[0]]))])
    test_y = np.array([[test_y[attributes[0]][i], test_y[attributes[1]][i], test_y[attributes[2]][i]] for i in range(len(test_y[attributes[0]]))])

    x = [np.array(x[i:i + batch_size]) for i in range(0, len(x), batch_size)]
    dev_x = [np.array(dev_x[i:i + batch_size]) for i in range(0, len(dev_x), batch_size)]
    test_x = [np.array(test_x[i:i + batch_size]) for i in range(0, len(test_x), batch_size)]

    loss_wts = data.loc[:, [(attr_conf[attr] + ".norm") for attr in attributes]].values
    loss_wts = [np.array(loss_wts[i:i + batch_size]) for i in range(0, len(loss_wts), batch_size)]
    dev_loss_wts = data_dev_mean.loc[:, [(attr_conf[attr] + ".norm") for attr in attributes]].values
    test_loss_wts = data_test_mean.loc[:, [(attr_conf[attr] + ".norm") for attr in attributes]].values

    if search_on:
        search_accs = []
        for (hidden_state, alpha, drp, act) in all_grid_params:
            clf = simpleMLP(device=device, input_size=x[0].shape[1],
                    layers=hidden_state,
                    output_size=y[0].shape[1], p_dropout=drp,
                    activation=act)
            clf.to(device)
            loss_function = BCELoss(reduction="none")
            parameters = [p for p in clf.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(parameters, weight_decay=alpha)
            early_stopping = [0]
            for epoch in range(20):
                for x_, dev_x_, y_, wts in zip(x, dev_x, y, loss_wts):
                    optimizer.zero_grad()

                    x_ = torch.tensor(x_, dtype=torch.float, device=device)
                    dev_x_ = torch.tensor(dev_x_, dtype=torch.float, device=device)
                    y_ = torch.tensor(y_, dtype=torch.float, device=device)
                    wts = torch.tensor(wts, dtype=torch.float, device=device)

                    y_pred = clf(x_)
                    loss = loss_function(y_pred, y_)
                    loss = torch.sum(loss * wts) / batch_size
                    loss.backward()
                    optimizer.step()
                    # loss_trace.append(float(loss.data))
                clf = clf.eval()
                y_pred_dev = predict(clf, dev_x, device)
                clf = clf.train()
                acc = accuracy(dev_y, y_pred_dev)
                early_stopping.append(acc)
                if early_stopping[-1] - early_stopping[-2] < 0:
                    search_accs.append((hidden_state, alpha, drp, act, early_stopping[-2]))
                    break
                else:
                    name_of_model = (prot + "_" +
                             "elmo:" + str(elmo_on) + "_" +
                             "glove:" + str(glove_on) + "_" +
                             "token:" + str(token_on) + "_" +
                             "type:" + str(type_on) + "_" +
                             "tokenabl:" + str(token_abl_names[tokenabl]) +
                             "_" + "abl:" + str(abl_dict[abl]) + "_" +
                             str(hidden_state) + "_" +
                             str(alpha) + "_" +
                             str(drp) + "_" + str(act))
                    Path = expanduser('~') + "/Desktop/saved_models/" + name_of_model
                    torch.save(clf.state_dict(), Path)
        print(max(search_accs, key=lambda x: x[-1]), "\n")
    else:
        hidden_state, alpha, drp, act = ((256, 64), 0.001, 0.4, 'relu')
        clf = simpleMLP(device=device, input_size=dev_x[0].shape[1],
                        layers=hidden_state, p_dropout=drp,
                        activation=act)
        clf.to(device)
        name_of_model = (prot + "_" + "elmo:" + str(elmo_on) + "_" +
                        "glove:" + str(glove_on) + "_" +
                        "token:" + str(token_on) + "_" +
                        "type:" + str(type_on) + "_" +
                        "tokenabl:" + str(token_abl_names[tokenabl]) + "_" +
                         "abl:" + str(abl_dict[abl]) + "_" +
                         str(hidden_state) + "_" +
                         str(alpha) + "_" +
                         str(drp) + "_" + str(act))
        best_model = expanduser('~') + "/Desktop/saved_models/" + name_of_model
        clf.load_state_dict(torch.load(best_model))
        clf.eval()
        if not test_on:
            y_pred_dev = predict(clf, dev_x, device)
            print_metrics(attributes, attr_map, attr_conf, dev_loss_wts, dev_y, y_pred_dev)
        else:
            y_pred_test = predict(clf, test_x, device)
            print_metrics(attributes=attributes, attr_map=attr_map,
                          attr_conf=attr_conf, wts=test_loss_wts,
                          y_true=test_y, y_pred=y_pred_test)


def predict(clf, x, device):
    predictions = np.empty((0, 3), int)
    for mb in x:
        mb = torch.tensor(mb, dtype=torch.float, device=device)
        preds = clf(mb)
        preds = preds > 0.5
        predictions = np.concatenate([predictions, preds.detach().cpu().numpy()])
    return predictions


def train():
    pass


def print_metrics(attributes, attr_map, attr_conf, wts, y_true, y_pred):
    print("Micro F1:", np.round(f1(y_true, y_pred, average='micro'), sigdig), "\nMacro F1:", np.round(f1(y_true, y_pred, average='macro'), sigdig),
        "\nTotal accuracy:", np.round(accuracy(y_true, y_pred), sigdig))
    for ind, attr in enumerate(attributes):
        print(attr_map[attr])
        # mode_ = mode(y_true[:, ind])[0][0]
        print("Accuracy :\t", np.round(accuracy(y_true[:, ind], y_pred[:, ind]), sigdig), np.round(accuracy(y_true[:, ind], y_pred[:, ind], sample_weight=wts[:, ind]), sigdig), "\n",
        "Precision :\t", np.round(precision(y_true[:, ind], y_pred[:, ind]), sigdig), np.round(precision(y_true[:, ind], y_pred[:, ind], sample_weight=wts[:, ind]), sigdig), "\n",
        "Recall :\t", np.round(recall(y_true[:, ind], y_pred[:, ind]), sigdig), np.round(recall(y_true[:, ind], y_pred[:, ind], sample_weight=wts[:, ind]), sigdig), "\n",
        "F1 score: \t", np.round(f1(y_true[:, ind], y_pred[:, ind]), sigdig), np.round(f1(y_true[:, ind], y_pred[:, ind], sample_weight=wts[:, ind]), sigdig), "\n")


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
    parser.add_argument('--abl',
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
    parser.add_argument('--test',
                        action='store_true',
                        help='Run test')

    sigdig = 3
    args = parser.parse_args()
    home = expanduser('~')

    main(prot=args.prot, batch_size=args.batch_size, glove_on=args.glove,
         elmo_on=args.elmo, token_on=args.token, type_on=args.type,
         tokenabl=args.tokenabl, abl=args.abl, search_on=args.search,
         test_on=args.test)
