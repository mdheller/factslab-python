import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score as accuracy, precision_score as precision, recall_score as recall, f1_score as f1
from sklearn.utils import shuffle
from predpatt import load_conllu, PredPatt, PredPattOpts
from os.path import expanduser
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from collections import Counter
from factslab.utility import ridit, dev_mode_group
from scipy.stats import mode
import warnings
import pickle
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, framenet, verbnet
import sys
warnings.filterwarnings("ignore", category=DeprecationWarning)


def concreteness_score(concreteness, lemma):
    '''
        Returns concreteness score(float) for lemma
    '''
    if lemma in concreteness['Word'].values.tolist():
        score = concreteness[concreteness['Word'] == lemma]['Conc.M'].values[0]
    elif lemma.lower() in concreteness['Word'].values.tolist():
        score = concreteness[concreteness['Word'] == lemma.lower()]['Conc.M'].values[0]
    else:
        score = 2.5

    return score


def lcs_score(lcs, lemma):
    '''
        Returns eventiveness score(0 or 1)
    '''
    if lemma in lcs.verbs:
        if True in lcs.eventive(lemma):
            score = 1
        else:
            score = 0
    else:
        score = -1

    return score


def features_func(sent_feat, token, lemma, dict_feats, prot, concreteness, lcs):
    lmt = WordNetLemmatizer()
    '''Extract features from a word'''
    sent = sent_feat[0]
    feats = sent_feat[1]
    deps = [x[2] for x in sent.tokens[token].dependents]
    deps_text = [x[2].text for x in sent.tokens[token].dependents]
    deps_feats = '|'.join([(a + "_dep") for x in deps for a in feats[x.position].split('|')])

    all_feats = (feats[token] + '|' + deps_feats).split('|')
    all_feats = list(filter(None, all_feats))
    # UD Lexical features
    for f in all_feats:
        dict_feats[f] += 1

    # Lexical item features
    for f in deps_text:
        if f in dict_feats.keys():
            dict_feats[f] += 1

    # wordnet supersense of lemma
    for synset in wordnet.synsets(lemma):
        dict_feats[synset.lexname()] = 1

    # # framenet name
    # for f_name in [x.name for x in framenet.frames_by_lemma(lemma)]:
    #     dict_feats[f_name] = 1

    # Predicate features
    if prot == "pred":
        # verbnet class
        f_lemma = verbnet.classids(lemma=lemma)
        for f in f_lemma:
            dict_feats[f] = 1

        # lcs eventiveness
        dict_feats['lcs'] = lcs_score(lcs, lemma)

        # sum of concreteness score of dependents
        dict_feats['concreteness'] = 0
        for g_lemma in [lmt.lemmatize(x[2].text) for x in sent.tokens[token].dependents]:
            dict_feats['concreteness'] += concreteness_score(concreteness, g_lemma)
        if len(sent.tokens[token].dependents):
            dict_feats['concreteness'] /= len(sent.tokens[token].dependents)
        else:
            dict_feats['concreteness'] = 2.5
    # Argument features
    else:
        dict_feats['concreteness'] = concreteness_score(concreteness, lemma)

        # lcs eventiveness score and verbnet class of argument head
        if not sent.tokens[token].gov:
            dict_feats['lcs'] = -1
        else:
            gov_lemma = lmt.lemmatize(sent.tokens[token].gov.text, 'v')
            dict_feats['lcs'] = lcs_score(lcs, gov_lemma)

            for f_lemma in verbnet.classids(lemma=gov_lemma):
                dict_feats[f_lemma] += 1

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
                        default='mlp')
    parser.add_argument('--load_data',
                        action='store_true')

    sigdig = 3
    # parse arguments
    args = parser.parse_args()
    home = expanduser('~')

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

    if args.load_data:
        data = pd.read_csv(datafile, sep="\t")
        data = data.dropna()
        data['Unique.ID'] = data.apply(lambda x: x['Split'] + " sent_" + str(x['Sentence.ID']) + "_" + str(x["Span"]), axis=1)
        data['Split.Sentence.ID'] = data.apply(lambda x: x['Split'] + " sent_" + str(x['Sentence.ID']), axis=1)

        # Load the sentences
        sentences = {}
        with open(home + '/Desktop/protocols/data/sentences.tsv', 'r') as f:
            for line in f.readlines():
                structs = line.split('\t')
                sentences[structs[0]] = structs[1].split()
        data['Sentence'] = data['Split.Sentence.ID'].map(lambda x: sentences[x])

        # Load the features
        features = {}
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
        all_feats = '|'.join(['|'.join(all_x[i][1]) for i in range(len(all_x))])
        feature_cols = Counter(all_feats.split('|'))
        dict_feats = {}

        f = open('concrete.pkl', 'rb')
        concreteness = pickle.load(f)
        f.close()

        from lcsreader import LexicalConceptualStructureLexicon
        lcs = LexicalConceptualStructureLexicon('verbs-English.lcs')

        # Wordnet supersenses(lexicographer names)
        supersenses = list(set([x.lexname() for x in wordnet.all_synsets()]))

        # # Framenet
        # frame_names = [x.name for x in framenet.frames()]

        lexical_feats = ['can', 'could', 'should', 'would', 'will', 'may', 'might', 'must', 'ought', 'dare', 'need'] + ['the', 'an', 'a', 'few', 'another', 'some', 'many', 'each', 'every', 'this', 'that', 'any', 'most', 'all', 'both', 'these']

        for f in verbnet.classids() + lexical_feats + supersenses:
            dict_feats[f] = 0
        for a in feature_cols.keys():
            dict_feats[a] = 0
            dict_feats[a + "_dep"] = 0

        x_pd = pd.DataFrame([features_func(sent_feat=sent, token=token, lemma=lemma, dict_feats=dict_feats.copy(), prot=args.prot, concreteness=concreteness, lcs=lcs) for sent, token, lemma in zip(raw_x, tokens, lemmas)])
        y = {}
        for attr in attributes:
            y[attr] = data[attr_map[attr] + ".norm"].values

        dev_x_pd = pd.DataFrame([features_func(sent_feat=sent, token=token, lemma=lemma, dict_feats=dict_feats.copy(), prot=args.prot, concreteness=concreteness, lcs=lcs) for sent, token, lemma in zip(raw_dev_x, dev_tokens, dev_lemmas)])

        # Figure out which columns to drop(they're always zero)
        todrop1 = dev_x_pd.columns[(dev_x_pd == 0).all()].values.tolist()
        todrop = x_pd.columns[(x_pd == 0).all()].values.tolist()
        intdrop = [a for a in todrop if a not in todrop1]
        cols_to_drop = cols_to_drop = list(set(todrop) - set(intdrop))

        x = x_pd.drop(cols_to_drop, axis=1).values
        dev_x = dev_x_pd.drop(cols_to_drop, axis=1).values
        dev_y = {}
        for attr in attributes:
            dev_y[attr] = data_dev_mean[attr_map[attr] + ".norm"].values

        fout = open(args.prot + 'baseline.pkl', 'wb')
        pickle.dump([x, y, dev_x, dev_y, data, data_dev_mean], fout)
        fout.close()
        sys.exit(0)

    fin = open(args.prot + 'baseline.pkl', 'rb')
    x, y, dev_x, dev_y, data, data_dev_mean = pickle.load(fin)

    if args.model == "mlp":
        classifier = MLPClassifier()

        parameters = {'hidden_layer_sizes': [(256, 32), (512, 64), (512, 32)], 'alpha': [0.0001, 0.001, 0.01, 0.1], 'early_stopping': [True], 'activation': ['tanh', 'relu']}

        y = np.array([[y[attributes[0]][i], y[attributes[1]][i], y[attributes[2]][i]] for i in range(len(y[attributes[0]]))])
        dev_y = np.array([[dev_y[attributes[0]][i], dev_y[attributes[1]][i], dev_y[attributes[2]][i]] for i in range(len(dev_y[attributes[0]]))])

        all_x = np.concatenate((x, dev_x), axis=0)
        all_y = np.concatenate((y, dev_y), axis=0)
        test_fold = [-1 for i in range(len(x))] + [0 for j in range(len(dev_x))]
        ps = PredefinedSplit(test_fold=test_fold)
        clf = GridSearchCV(classifier, parameters, n_jobs=-1, verbose=1, cv=ps)
        clf.fit(all_x, all_y)
        print(clf.best_params_)
        # for p, tr in zip(clf.cv_results_['params'], clf.cv_results_['mean_test_score']):
        #         print(p, np.round(tr, sigdig))

        y_pred_dev = clf.predict(dev_x)
        for ind, attr in enumerate(attributes):
            print(attr_map[attr])
            mode_ = mode(dev_y[:, ind])[0][0]
            print("Accuracy mode =", mode_, ":", np.round(accuracy(dev_y[:, ind], [mode_ for a in range(len(dev_y[:, ind]))]), sigdig))
            print("Accuracy :\t", np.round(accuracy(dev_y[:, ind], y_pred_dev[:, ind]), sigdig), "\n",
                  "Precision :\t", np.round(precision(dev_y[:, ind], y_pred_dev[:, ind], pos_label=mode_), sigdig), "\n",
                  "Recall :\t", np.round(recall(dev_y[:, ind], y_pred_dev[:, ind], pos_label=mode_), sigdig), "\n",
                  "F1 score: \t", np.round(f1(dev_y[:, ind], y_pred_dev[:, ind], pos_label=mode_), sigdig), "\n")
    else:
        if args.model == "lr":
            classifier = LR()
            parameters = {'C': [0.01, 0.1, 0.5, 1, 10, 100], 'penalty': ['l1', 'l2']}
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
            clf = GridSearchCV(classifier, parameters, n_jobs=-1, verbose=1, cv=ps, fit_params={'sample_weight': all_loss_wts})
            clf.fit(all_x, all_y)

            y_pred_dev = clf.predict(dev_x)
            print(attr_map[attr])
            mode_ = mode(dev_y[attr])[0][0]
            print("Accuracy mode =", mode_, ":", np.round(accuracy(dev_y[attr], [mode_ for a in range(len(dev_y[attr]))]), sigdig))
            print("Accuracy :\t", np.round(accuracy(dev_y[attr], y_pred_dev), sigdig), "\n",
                  "Precision :\t", np.round(precision(dev_y[attr], y_pred_dev, pos_label=mode_), sigdig), "\n",
                  "Recall :\t", np.round(recall(dev_y[attr], y_pred_dev, pos_label=mode_), sigdig), "\n",
                  "F1 score: \t", np.round(f1(dev_y[attr], y_pred_dev, pos_label=mode_), sigdig), "\n")
