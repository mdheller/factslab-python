import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score as accuracy, precision_score as precision, recall_score as recall
from predpatt import load_conllu
from predpatt import PredPatt
from predpatt import PredPattOpts
from os.path import expanduser
from statistics import median
from sklearn.model_selection import GridSearchCV
from collections import Counter
from factslab.utility import ridit, dev_mode_group

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def features_func(sent_feat, token, dict_feats):
    '''Extract features from a word'''
    sent = sent_feat[0]
    feats = sent_feat[1]
    deps = [x[2] for x in sent.tokens[token].dependents]
    deps_text = [x[2].text for x in sent.tokens[token].dependents]
    deps_feats = '|'.join([(a + "_dep") for x in deps for a in feats[x.position].split('|')])
    # Not creating separate features for PronType=Art and Poss=yes, using existing columns because its rare that they'll be in the token itself
    all_feats = (feats[token] + '|' + deps_feats).split('|')
    all_feats = list(filter(None, all_feats))
    for f in all_feats:
        dict_feats[f] += 1
    for f in deps_text:
        if f in dict_feats.keys():
            dict_feats[f] += 1
    return dict_feats


if __name__ == "__main__":
    # initialize argument parser
    description = 'Run an RNN regression on Genericity protocol annotation.'
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--prot',
                        type=str,
                        default="noun")
    parser.add_argument('--model',
                        type=str,
                        default='lr')

    sigdig = 3
    # parse arguments
    args = parser.parse_args()

    if args.model == "lr":
        classifier = LR()
    else:
        classifier = SVC()
    if args.prot == "noun":
        datafile = "../../../protocols/data/arg_long_data.tsv"
        attributes = ["part", "kind", "abs"]
        attr_map = {"part": "Is.Particular", "kind": "Is.Kind", "abs": "Is.Abstract"}
        attr_conf = {"part": "Part.Confidence", "kind": "Kind.Confidence",
                 "abs": "Abs.Confidence"}
    else:
        datafile = "../../../protocols/data/pred_long_data.tsv"
        attributes = ["part", "hyp", "dyn"]
        attr_map = {"part": "Is.Particular", "dyn": "Is.Dynamic", "hyp": "Is.Hypothetical"}
        attr_conf = {"part": "Part.Confidence", "dyn": "Dyn.Confidence",
                 "hyp": "Hyp.Confidence"}

    data = pd.read_csv(datafile, sep="\t")

    data['Unique.ID'] = data.apply(lambda x: x['Split'] + " sent_" + str(x['Sentence.ID']) + "_" + str(x["Span"]), axis=1)
    data['Split.Sentence.ID'] = data.apply(lambda x: x['Split'] + " sent_" + str(x['Sentence.ID']), axis=1)

    # Load the features
    features = {}

    # Don't read_csv the features file. read_csv can't handle quotes
    with open("features.tsv", 'r') as f:
        for line in f.readlines():
            feats = line.split('\t')
            features[feats[0]] = feats[1].split()

    # Load the predpatt objects for creating features

    files = ['/UD_English-r1.2/en-ud-train.conllu',
             '/UD_English-r1.2/en-ud-dev.conllu',
             '/UD_English-r1.2/en-ud-test.conllu']
    home = expanduser("~/Downloads/")
    options = PredPattOpts(resolve_relcl=True, borrow_arg_for_relcl=True, resolve_conj=False, cut=True)  # Resolve relative clause
    patt = {}

    for file in files:
        path = home + file
        with open(path, 'r') as infile:
            for sent_id, ud_parse in load_conllu(infile.read()):
                patt[file[23:][:-7] + " " + sent_id] = PredPatt(ud_parse, opts=options)

    data['Structure'] = data['Split.Sentence.ID'].map(lambda x: (patt[x], features[x]))

    # Split the datasets into train, dev, test
    data_test = data[data['Split'] == 'test']
    data_dev = data[data['Split'] == 'dev']
    data = data[data['Split'] == 'train']

    # Ridit scoring annotations and confidence ratings
    for attr in attributes:
        resp = attr_map[attr]
        resp_conf = attr_conf[attr]
        data[resp_conf + ".norm"] = data.groupby('Annotator.ID')[resp_conf].transform(ridit)
        data[resp + ".norm"] = data[resp].map(lambda x: 1 if x else -1)
        data_dev[resp_conf + ".norm"] = data_dev.groupby('Annotator.ID')[resp_conf].transform(ridit)
        data_dev[resp + ".norm"] = data_dev[resp].map(lambda x: 1 if x else -1)

    # Shuffle the data
    data = data.sample(frac=1).reset_index(drop=True)
    data_dev = data_dev.sample(frac=1).reset_index(drop=True)
    data_test = data_test.sample(frac=1).reset_index(drop=True)

    raw_x = data['Structure'].tolist()
    tokens = data['Root.Token'].tolist()

    data_dev_mean = data_dev.groupby('Unique.ID', as_index=False).apply(lambda x: dev_mode_group(x, attributes, attr_map, attr_conf)).reset_index(drop=True)

    raw_dev_x = data_dev_mean['Structure'].tolist()
    dev_tokens = data_dev_mean['Root.Token'].tolist()

    all_x = raw_x + raw_dev_x
    # Now figure out the features here before calculating them
    all_feats = '|'.join(['|'.join(all_x[i][1]) for i in range(len(all_x))])
    feature_cols = Counter(all_feats.split('|'))
    dict_feats = {}
    # Create dictionary of features for root and dependents
    for a in feature_cols.keys():
        dict_feats[a] = 0
        dict_feats[a + "_dep"] = 0
    if args.prot == "noun":
        lexical_feats = ['the', 'an', 'a', 'few', 'another', 'some', 'many', 'each', 'every', 'this', 'that', 'any', 'most', 'all', 'both', 'these']
    else:
        lexical_feats = ['can', 'could', 'should', 'would', 'will', 'may', 'might', 'must', 'ought', 'dare', 'need']
    for new_key in lexical_feats:
        dict_feats[new_key] = 0
    x = []
    x_pd = pd.DataFrame([features_func(sent, token, dict_feats.copy()) for sent, token in zip(raw_x, tokens)])
    x = x_pd.values
    y = {}
    for attr in attributes:
        y[attr] = data[attr_map[attr] + ".norm"].values

    dev_x = pd.DataFrame([features_func(sent, token, dict_feats.copy()) for sent, token in zip(raw_dev_x, dev_tokens)]).values
    dev_y = {}
    parameters = {}
    for attr in attributes:
        dev_y[attr] = data_dev_mean[attr_map[attr] + ".norm"].values
        if args.model == "lr":
            parameters[attr] = {'C': [1, 10, 100], 'penalty': ['l1', 'l2']}
        else:
            parameters[attr] = {'C': [1, 2], 'gamma': [0.25, 0.5, 1], 'kernel': ['linear', 'rbf']}

    for attr in attributes:
        print(attr_map[attr])
        mode_ = data_dev_mean[attr_map[attr] + ".norm"].mode()
        # Grid search 
        loss_wts = data[attr_conf[attr] + ".norm"].values
        print("Accuracy with mode:", np.round(accuracy(dev_y[attr], [mode_ for a in range(len(dev_y[attr]))]), sigdig))
        clf = GridSearchCV(classifier, parameters[attr], return_train_score=True, n_jobs=10, verbose=1, fit_params={'sample_weight': loss_wts})
        clf.fit(x, y[attr])

        for p, tr, trr in zip(clf.cv_results_['params'], clf.cv_results_['mean_test_score'], clf.cv_results_['mean_train_score']):
            print(p, tr, trr)

        print("Accuracy on DEV")
        print(clf.score(dev_x, dev_y[attr]))
