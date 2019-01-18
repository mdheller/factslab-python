import argparse
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score as accuracy, precision_score as precision, recall_score as recall, f1_score as f1, mean_absolute_error as mae, r2_score as r2
from sklearn.utils import shuffle
from predpatt import load_conllu, PredPatt, PredPattOpts
from os.path import expanduser
from sklearn.model_selection import GridSearchCV, PredefinedSplit, RandomizedSearchCV
from collections import Counter, defaultdict
from factslab.utility import ridit, dev_mode_group, get_elmo, load_glove_embedding
from scipy.stats import mode, uniform
import pickle
from nltk.corpus import wordnet, framenet, verbnet
import sys
import joblib
import random

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)


if __name__ == "__main__":
    # initialize argument parser
    description = 'Run an RNN regression on Genericity protocol annotation.'
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--prot',
                        type=str,
                        default='arg')
    parser.add_argument('--batch_size',
                        type=int,
                        default=128)
    parser.add_argument('--abl',
                        type=int,
                        default=0)
    parser.add_argument('--tokenabl',
                        type=int,
                        default=0)
    parser.add_argument('--model',
                        type=str,
                        default='mlp')
    parser.add_argument('--modeltype',
                        type=str,
                        default='regression')
    parser.add_argument('--hand',
                        action='store_true',
                        help='Turn on hand engineering feats')
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

    framnet_posdict = {'V': 'VERB', 'N': 'NOUN', 'A': 'ADJ', 'ADV': 'ADV', 'PREP': 'ADP', 'NUM': 'NUM', 'INTJ': 'INTJ', 'ART': 'DET', 'C': 'CCONJ', 'SCON': 'SCONJ', 'PRON': 'PRON', 'IDIO': 'X', 'AVP': 'ADV'}
    sigdig = 3
    # parse arguments
    args = parser.parse_args()
    home = expanduser('~')

    if args.prot == "arg":
        datafile = home + '/Desktop/protocols/data/noun_raw_data_norm_122218.tsv'
        attributes = ["part", "kind", "abs"]
        attr_map = {"part": "Is.Particular", "kind": "Is.Kind", "abs": "Is.Abstract"}
        attr_conf = {"part": "Part.Confidence", "kind": "Kind.Confidence",
                 "abs": "Abs.Confidence"}
    else:
        datafile = home + '/Desktop/protocols/data/pred_raw_data_norm_122218.tsv'
        attributes = ["part", "hyp", "dyn"]
        attr_map = {"part": "Is.Particular", "dyn": "Is.Dynamic", "hyp": "Is.Hypothetical"}
        attr_conf = {"part": "Part.Confidence", "dyn": "Dyn.Confidence",
                 "hyp": "Hyp.Confidence"}

    fin = open('data/' + args.prot + 'hand.pkl', 'rb')
    x_pd, y, dev_x_pd, dev_y, data, data_dev_mean, feature_names, cols_to_drop = pickle.load(fin)
    fin.close()

    verbnet_classids, supersenses, frame_names, lcs_feats, conc_cols, lexical_feats, all_ud_feature_cols = feature_names
    abl_dict = {0: [], 1: verbnet_classids, 2: supersenses, 3: frame_names, 4: lcs_feats, 5: conc_cols, 6: lexical_feats, 7: all_ud_feature_cols}
    type_cols = verbnet_classids + supersenses + frame_names + lcs_feats + conc_cols
    token_cols = lexical_feats + all_ud_feature_cols
    abl_names = {0: 'None', 1: 'verbnet', 2: 'supersenses', 3: 'frames', '4': 'lcs', 5: 'conc', 6: 'lexical', 7: 'UD'}
    token_abl_names = {0: 'None', 1: 'UPOS=', 2: 'XPOS=', 3: 'mood', 4: 'DEPRE', 5: 'U+X'}

    tr_elmo = open('data/' + args.prot + 'train_elmo.pkl', 'rb')
    x_elmo = pickle.load(tr_elmo)
    tr_elmo.close()

    dev_elmo = open('data/' + args.prot + 'dev_elmo.pkl', 'rb')
    dev_x_elmo = pickle.load(dev_elmo)
    dev_elmo.close()

    tr_glove = open('data/' + args.prot + 'train_glove.pkl', 'rb')
    x_glove = pickle.load(tr_glove)
    tr_glove.close()

    dev_glove = open('data/' + args.prot + 'dev_glove.pkl', 'rb')
    dev_x_glove = pickle.load(dev_glove)
    dev_glove.close()

    if args.type and not args.token and not args.abl:
        x = x_pd.drop(x_pd.columns.intersection(token_cols), axis=1).values
        dev_x = dev_x_pd.drop(dev_x_pd.columns.intersection(token_cols), axis=1).values
    elif not args.type and args.token and not args.abl:
        x = x_pd.drop(x_pd.columns.intersection(type_cols), axis=1).values
        dev_x = dev_x_pd.drop(dev_x_pd.columns.intersection(type_cols), axis=1).values
    elif args.abl:
        ablation = abl_dict[args.abl]
        x = x_pd.drop(x_pd.columns.intersection(ablation), axis=1).values
        dev_x = dev_x_pd.drop(dev_x_pd.columns.intersection(ablation), axis=1).values
    elif args.tokenabl:
        if args.tokenabl == 3:
            ud_feats_to_remove = [a for a in all_ud_feature_cols if a[0:5] not in ['UPOS=', 'XPOS=', 'DEPRE']]
        elif args.tokenabl == 5:
            ud_feats_to_remove = [a for a in all_ud_feature_cols if a[0:5] in ['UPOS=', 'XPOS=']]
        else:
            ud_feats_to_remove = [a for a in all_ud_feature_cols if a[0:5] == token_abl_names[args.tokenabl]]
        ablation = type_cols + lexical_feats + ud_feats_to_remove
        x = x_pd.drop(x_pd.columns.intersection(ablation), axis=1).values
        dev_x = dev_x_pd.drop(dev_x_pd.columns.intersection(ablation), axis=1).values
    else:
        x = x_pd.values
        dev_x = dev_x_pd.values

    if args.hand and not args.elmo and not args.glove:
        pass
    elif not args.hand and args.elmo and not args.glove:
        x = x_elmo
        dev_x = dev_x_elmo
    elif not args.hand and not args.elmo and args.glove:
        x = x_glove
        dev_x = dev_x_glove
    elif args.hand and args.elmo and not args.glove:
        x = np.concatenate((x, x_elmo), axis=1)
        dev_x = np.concatenate((dev_x, dev_x_elmo), axis=1)
    elif args.hand and args.elmo and args.glove:
        x = np.concatenate((x, x_elmo, x_glove), axis=1)
        dev_x = np.concatenate((dev_x, dev_x_elmo, dev_x_glove), axis=1)
    elif not args.hand and args.elmo and args.glove:
        x = np.concatenate((x_elmo, x_glove), axis=1)
        dev_x = np.concatenate((dev_x_elmo, dev_x_glove), axis=1)
    elif args.hand and not args.elmo and args.glove:
        x = np.concatenate((x, x_glove), axis=1)
        dev_x = np.concatenate((dev_x, dev_x_glove), axis=1)
    else:
        sys.exit('Choose a represenation for x')

    if args.model == "mlp":
        classifier = MLPClassifier()
        random_state = np.random.RandomState(0)
        grid_params = {'hidden_layer_sizes': [(512, 256), (512, 128), (512, 64), (512, 32), (512,), (256, 128), (256, 64), (256, 32), (256,), (128, 64), (128, 32), (128), (64, 32), (64,), (32,)], 'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1], 'early_stopping': [True], 'batch_size': [32], 'random_state': [random_state]}
        best_params = {'hidden_layer_sizes': (512, 32), 'alpha': 0.001, 'early_stopping': True, 'activation': 'relu', 'batch_size': 32, 'random_state': random_state}
        best_grid_params = {'hidden_layer_sizes': [(512, 32)], 'alpha': [0.0001], 'early_stopping': [True], 'activation': ['relu'], 'batch_size': [32]}

        y = np.array([[y[attributes[0]][i], y[attributes[1]][i], y[attributes[2]][i]] for i in range(len(y[attributes[0]]))])
        dev_y = np.array([[dev_y[attributes[0]][i], dev_y[attributes[1]][i], dev_y[attributes[2]][i]] for i in range(len(dev_y[attributes[0]]))])

        all_x = np.concatenate((x, dev_x), axis=0)
        all_y = np.concatenate((y, dev_y), axis=0)
        test_fold = [-1 for i in range(len(x))] + [0 for j in range(len(dev_x))]
        ps = PredefinedSplit(test_fold=test_fold)
        best_results = []
        if args.search:
            clf = GridSearchCV(classifier, grid_params, n_jobs=-1, verbose=1, cv=ps, refit=False)
            clf.fit(all_x, all_y)
            print(clf.best_params_)
            # for p, tr in zip(clf.cv_results_['params'], clf.cv_results_['mean_test_score']):
            #     print(p['alpha'], p['hidden_layer_sizes'], np.round(tr, sigdig))
            print("============================================")
        else:
            clf = classifier.set_params(**best_params)
            clf.fit(x, y)
            y_pred_dev = clf.predict(dev_x)
            print(np.round(f1(dev_y, y_pred_dev, average='micro'), sigdig), np.round(f1(dev_y, y_pred_dev, average='macro'), sigdig))
            for ind, attr in enumerate(attributes):
                print(attr_map[attr])
                mode_ = mode(dev_y[:, ind])[0][0]
                print("Accuracy :\t", np.round(accuracy(dev_y[:, ind], y_pred_dev[:, ind]), sigdig), np.round(accuracy(dev_y[:, ind], y_pred_dev[:, ind], sample_weight=data_dev_mean[attr_conf[attr] + ".norm"].values), sigdig), "\n",
                "Precision :\t", np.round(precision(dev_y[:, ind], y_pred_dev[:, ind]), sigdig), np.round(precision(dev_y[:, ind], y_pred_dev[:, ind], sample_weight=data_dev_mean[attr_conf[attr] + ".norm"].values), sigdig), "\n",
                "Recall :\t", np.round(recall(dev_y[:, ind], y_pred_dev[:, ind]), sigdig), np.round(recall(dev_y[:, ind], y_pred_dev[:, ind], sample_weight=data_dev_mean[attr_conf[attr] + ".norm"].values), sigdig), "\n",
                "F1 score: \t", np.round(f1(dev_y[:, ind], y_pred_dev[:, ind]), sigdig), np.round(f1(dev_y[:, ind], y_pred_dev[:, ind], sample_weight=data_dev_mean[attr_conf[attr] + ".norm"].values), sigdig), "\n")
    else:
        from sklearn.linear_model import LogisticRegression as LR
        from sklearn.svm import LinearSVC
        if args.model == "lr":
            classifier = LR()
            parameters = {'C': uniform(loc=1, scale=1), 'penalty': ['l1', 'l2']}
            best_params = {'C': 1.05, 'penalty': 'l1'}
        elif args.model == "svm":
            classifier = LinearSVC()
            parameters = {'C': [0.01, 0.1, 0.5, 1, 10, 100]}

        all_x = np.concatenate((x, dev_x), axis=0)
        test_fold = [-1 for i in range(len(x))] + [0 for j in range(len(dev_x))]
        ps = PredefinedSplit(test_fold=test_fold)

        for attr in attributes:
            loss_wts = data[attr_conf[attr] + ".norm"].values
            dev_loss_wts = data_dev_mean[attr_conf[attr] + ".norm"].values

            all_y = np.concatenate((y[attr], dev_y[attr]))
            all_loss_wts = np.concatenate((loss_wts, dev_loss_wts))
            if args.search:
                clf = RandomizedSearchCV(classifier, parameters, n_jobs=-3, verbose=1, cv=ps, fit_params={'sample_weight': all_loss_wts})
                clf.fit(all_x, all_y)
                print(clf.best_params_)
            else:
                clf = classifier.set_params(**best_params)
                clf.fit(x, y[attr])

            y_pred_dev = clf.predict(dev_x)
            print(attr_map[attr])
            mode_ = mode(dev_y[attr])[0][0]
            print("Accuracy mode =", mode_, ":", np.round(accuracy(dev_y[attr], [mode_ for a in range(len(dev_y[attr]))]), sigdig))
            print("Accuracy :\t", np.round(accuracy(dev_y[attr], y_pred_dev), sigdig), "\n",
                  "Precision :\t", np.round(precision(dev_y[attr], y_pred_dev, pos_label=mode_), sigdig), "\n",
                  "Recall :\t", np.round(recall(dev_y[attr], y_pred_dev, pos_label=mode_), sigdig), "\n",
                  "F1 score: \t", np.round(f1(dev_y[attr], y_pred_dev, pos_label=mode_), sigdig), "\n")
