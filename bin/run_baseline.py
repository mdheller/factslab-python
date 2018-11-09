import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score as accuracy, precision_score as precision, recall_score as recall, f1_score as f1
from sklearn.utils import shuffle
from predpatt import load_conllu
from predpatt import PredPatt
from predpatt import PredPattOpts
from os.path import expanduser
from sklearn.model_selection import GridSearchCV
from collections import Counter
from factslab.utility import ridit, dev_mode_group
from nltk.corpus import verbnet
from scipy.stats import mode
import warnings
import pickle
warnings.filterwarnings("ignore", category=DeprecationWarning)


def features_func(sent_feat, token, lemma, dict_feats, prot, concreteness=None, lcs=None):
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
    # now do lexical features
    for f in deps_text:
        if f in dict_feats.keys():
            dict_feats[f] += 1

    if prot == "pred":
        f_lemma = verbnet.classids(lemma=lemma)
        for f in f_lemma:
            dict_feats[f] = 1
        if lemma in lcs.verbs:
            if True in lcs.eventive(lemma):
                dict_feats['lcs'] = 1
            else:
                dict_feats['lcs'] = 0
        else:
            dict_feats['lcs'] = -1
    else:
        if lemma in concreteness['Word'].values.tolist():
            dict_feats['concreteness'] = concreteness[concreteness['Word'] == lemma.lower()]['Conc.M'].values[0]
        else:
            dict_feats['concreteness'] = 2.5

    return dict_feats


if __name__ == "__main__":
    # initialize argument parser
    description = 'Run an RNN regression on Genericity protocol annotation.'
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--prot',
                        type=str,
                        default='noun')
    parser.add_argument('--model',
                        type=str,
                        default='lr')
    parser.add_argument('--create',
                        action='store_true')

    sigdig = 3
    # parse arguments
    args = parser.parse_args()
    home = expanduser('~')
    if args.model == "lr":
        classifier = LR()
    else:
        classifier = LinearSVC()
    if args.prot == "noun":
        datafile = home + '/Desktop/protocols/data/arg_long_data.tsv'
        attributes = ["part", "kind", "abs"]
        attr_map = {"part": "Is.Particular", "kind": "Is.Kind", "abs": "Is.Abstract"}
        attr_conf = {"part": "Part.Confidence", "kind": "Kind.Confidence",
                 "abs": "Abs.Confidence"}
    else:
        datafile = home + '/Desktop/protocols/data/pred_long_data.tsv'
        attributes = ["part", "hyp", "dyn"]
        attr_map = {"part": "Is.Particular", "dyn": "Is.Dynamic", "hyp": "Is.Hypothetical"}
        attr_conf = {"part": "Part.Confidence", "dyn": "Dyn.Confidence",
                 "hyp": "Hyp.Confidence"}
    if args.create:
        data = pd.read_csv(datafile, sep="\t")
        data = data.dropna()
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
        data_test = data[data['Split'] == 'test'].reset_index(drop=True)
        data_dev = data[data['Split'] == 'dev'].reset_index(drop=True)
        data = data[data['Split'] == 'train'].reset_index(drop=True)

        # Ridit scoring annotations and confidence ratings
        for attr in attributes:
            resp = attr_map[attr]
            resp_conf = attr_conf[attr]
            data[resp_conf + ".norm"] = data.groupby('Annotator.ID')[resp_conf].transform(ridit)
            data_dev[resp_conf + ".norm"] = data_dev.groupby('Annotator.ID')[resp_conf].transform(ridit)
            data[resp + ".norm"] = data[resp].map(lambda x: 1 if x else 0)
            data_dev[resp + ".norm"] = data_dev[resp].map(lambda x: 1 if x else 0)

         # Shuffle the data
        data = shuffle(data).reset_index(drop=True)
        data_dev = shuffle(data_dev).reset_index(drop=True)
        data_test = shuffle(data_test).reset_index(drop=True)

        raw_x = data['Structure'].tolist()
        tokens = data['Root.Token'].tolist()
        lemmas = data['Lemma'].tolist()

        data_dev_mean = data_dev.groupby('Unique.ID', as_index=False).apply(lambda x: dev_mode_group(x, attributes, attr_map, attr_conf)).reset_index(drop=True)

        raw_dev_x = data_dev_mean['Structure'].tolist()
        dev_tokens = data_dev_mean['Root.Token'].tolist()
        dev_lemmas = data_dev_mean['Lemma'].tolist()

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
            lcs = None
            # concreteness ratings
            f = open('concrete.pkl', 'rb')
            concreteness = pickle.load(f)
            f.close()
        else:
            lexical_feats = ['can', 'could', 'should', 'would', 'will', 'may', 'might', 'must', 'ought', 'dare', 'need']
            concreteness = None
            # Verbnet classes
            for f in verbnet.classids():
                dict_feats[f] = 0
            # LCS eventiveness
            from lcsreader import LexicalConceptualStructureLexicon
            lcs = LexicalConceptualStructureLexicon('verbs-English.lcs')
        for new_key in lexical_feats:
            dict_feats[new_key] = 0

        x = []
        x_pd = pd.DataFrame([features_func(sent_feat=sent, token=token, lemma=lemma, dict_feats=dict_feats.copy(), prot=args.prot, concreteness=concreteness, lcs=lcs) for sent, token, lemma in zip(raw_x, tokens, lemmas)])
        x = x_pd.values
        y = {}
        for attr in attributes:
            y[attr] = data[attr_map[attr] + ".norm"].values

        dev_x = pd.DataFrame([features_func(sent_feat=sent, token=token, lemma=lemma, dict_feats=dict_feats.copy(), prot=args.prot, concreteness=concreteness, lcs=lcs) for sent, token, lemma in zip(raw_dev_x, dev_tokens, dev_lemmas)]).values
        dev_y = {}
        for attr in attributes:
            dev_y[attr] = data_dev_mean[attr_map[attr] + ".norm"].values

        fout = open(args.prot + 'baseline.pkl', 'wb')
        pickle.dump([x, y, dev_x, dev_y, data, data_dev_mean], fout)
        fout.close()
        import sys; sys.exit(0)
    else:
        fin = open(args.prot + 'baseline.pkl', 'rb')
        x, y, dev_x, dev_y, data, data_dev_mean = pickle.load(fin)
        parameters = {}
        for attr in attributes:
            if args.model == "lr":
                parameters[attr] = {'C': [0.1, 0.5, 1, 2, 5, 10, 100], 'penalty': ['l1', 'l2']}
            else:
                parameters[attr] = {'C': [0.01, 0.1, 0.5, 1, 10, 100]}

        for attr in attributes:
            print(attr_map[attr])
            mode_ = mode(dev_y[attr])[0][0]
            # Grid search
            loss_wts = data[attr_conf[attr] + ".norm"].values
            dev_loss_wts = data_dev_mean[attr_conf[attr] + ".norm"].values

            print("Accuracy with mode =", mode_, ":", np.round(accuracy(dev_y[attr], [mode_ for a in range(len(dev_y[attr]))]), sigdig))
            clf = GridSearchCV(classifier, parameters[attr], n_jobs=-1, verbose=0, fit_params={'sample_weight': loss_wts})
            clf.fit(x, y[attr])

            # for p, tr in zip(clf.cv_results_['params'], clf.cv_results_['mean_test_score']):
            #     print(p, np.round(tr, sigdig))

            y_pred_dev = clf.predict(dev_x)
            print("Accuracy :\t", np.round(accuracy(dev_y[attr], y_pred_dev), sigdig), "\n",
                  "Precision :\t", np.round(precision(dev_y[attr], y_pred_dev, pos_label=mode_), sigdig), "\n",
                  "Recall :\t", np.round(recall(dev_y[attr], y_pred_dev, pos_label=mode_), sigdig), "\n",
                  "F1 score: \t", np.round(f1(dev_y[attr], y_pred_dev, pos_label=mode_), sigdig), "\n")
