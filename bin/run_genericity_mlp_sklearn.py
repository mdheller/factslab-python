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
np.random.RandomState(0)


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
    parser.add_argument('--load_data',
                        action='store_true')
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

    if args.load_data:
        # sys.exit('no')
        data = pd.read_csv(datafile, sep="\t")
        data = data.dropna()
        # data['Unique.ID'] = data.apply(lambda x: x['Split'] + " sent_" + str(x['Sentence.ID']) + "_" + str(x["Span"]), axis=1)
        data['Split.Sentence.ID'] = data.apply(lambda x: x['Split'] + " sent_" + x['Sentence.ID'].split('_')[1], axis=1)

        # Load the sentences
        sentences = {}
        with open(home + '/Desktop/protocols/data/sentences.tsv', 'r') as f:
            for line in f.readlines():
                structs = line.split('\t')
                sentences[structs[0]] = structs[1].split()
        data['Sentences'] = data['Split.Sentence.ID'].map(lambda x: sentences[x])

        # Load the features
        features = {}
        upos = {}
        with open(home + '/Research/protocols/data/features-2.tsv', 'r') as f:
            for line in f.readlines():
                feats = line.split('\t')
                features[feats[0]] = [feats[1].split(), feats[2].split()]

        # Load the predpatt objects for creating features
        files = ['/Downloads/UD_English-r1.2/en-ud-train.conllu',
                 '/Downloads/UD_English-r1.2/en-ud-dev.conllu',
                 '/Downloads/UD_English-r1.2/en-ud-test.conllu']

        options = PredPattOpts(resolve_relcl=True, borrow_arg_for_relcl=True, resolve_conj=False, cut=True)  # Resolve relative clause
        patt = {}

        for file in files:
            path = home + file
            with open(path, 'r') as infile:
                for sent_id, ud_parse in load_conllu(infile.read()):
                    patt[file[33:][:-7] + " " + sent_id] = PredPatt(ud_parse, opts=options)

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
            data_test[resp_conf + ".norm"] = data_test.groupby('Annotator.ID')[resp_conf].transform(ridit)

            data[resp + ".ridit"] = ((data.groupby('Annotator.ID')[resp].transform(ridit)) * 2 - 1)
            data_dev[resp + ".ridit"] = ((data_dev.groupby('Annotator.ID')[resp].transform(ridit)) * 2 - 1)
            data_test[resp + ".ridit"] = ((data_test.groupby('Annotator.ID')[resp].transform(ridit)) * 2 - 1)

            if args.modeltype == "regression":
                data[resp + ".norm"] = data[resp + ".ridit"] * data[resp_conf + ".norm"]
                data_dev[resp + ".norm"] = data_dev[resp + ".ridit"] * data_dev[resp_conf + ".norm"]
                data_test[resp + ".norm"] = data_test[resp + ".ridit"] * data_test[resp_conf + ".norm"]
            else:
                data[resp + ".norm"] = data[resp].map(lambda x: 1 if x else 0)
                data_dev[resp + ".norm"] = data_dev[resp].map(lambda x: 1 if x else 0)
                data_test[resp + ".norm"] = data_test[resp].map(lambda x: 1 if x else 0)

        # Shuffle the data
        data = shuffle(data).reset_index(drop=True)
        data_dev = shuffle(data_dev).reset_index(drop=True)
        data_test = shuffle(data_test).reset_index(drop=True)

        # Create data in format amenable to extract features from
        raw_x = data['Structure'].tolist()
        tokens = data['Root.Token'].tolist()
        lemmas = data['Lemma'].tolist()
        sentences = data['Sentences'].tolist()

        data_dev_mean = data_dev.groupby('Unique.ID', as_index=False).apply(lambda x: dev_mode_group(x, attributes, attr_map, attr_conf, type="regression")).reset_index(drop=True)
        raw_dev_x = data_dev_mean['Structure'].tolist()
        dev_tokens = data_dev_mean['Root.Token'].tolist()
        dev_lemmas = data_dev_mean['Lemma'].tolist()
        dev_sentences = data_dev_mean['Sentences'].tolist()

        data_test_mean = data_test.groupby('Unique.ID', as_index=False).apply(lambda x: dev_mode_group(x, attributes, attr_map, attr_conf, type="regression")).reset_index(drop=True)
        raw_test_x = data_test_mean['Structure'].tolist()
        test_tokens = data_test_mean['Root.Token'].tolist()
        test_lemmas = data_test_mean['Lemma'].tolist()
        test_sentences = data_test_mean['Sentences'].tolist()

        all_x = raw_x
        all_feats = '|'.join(['|'.join(all_x[i][1][0]) for i in range(len(all_x))])
        feature_cols = Counter(all_feats.split('|'))
        list_of_all_upos = [[a.split('|')[0] for a in all_x[i][1][0]] for i in range(len(all_x))]
        list_of_all_lemmas = [all_x[i][1][1] for i in range(len(all_x))]
        # Lexical features
        lexical_feats = list(set(sum([[list_of_all_lemmas[i][j] for j in range(len(list_of_all_lemmas[i])) if list_of_all_upos[i][j] in ['UPOS=DET', 'UPOS=AUX']] for i in range(len(list_of_all_lemmas))], [])))

        # All UD dataset features
        all_ud_feature_cols = list(feature_cols.keys()) + [(a + "_dep") for a in feature_cols.keys()]

        # Concreteness
        concreteness = pd.read_csv(home + "/Desktop/protocols/data/concreteness.tsv", sep="\t")
        if args.prot == 'arg':
            conc_cols = ['concreteness']
        else:
            conc_cols = ['concreteness', 'max_conc', 'min_conc']
        f.close()

        # LCS eventivity
        from lcsreader import LexicalConceptualStructureLexicon
        lcs = LexicalConceptualStructureLexicon(home + '/Desktop/protocols/data/verbs-English.lcs')
        lcs_feats = ['lcs_eventive', 'lcs_stative']

        # Wordnet supersenses(lexicographer names)
        supersenses = list(set(['supersense=' + x.lexname() for x in wordnet.all_synsets()]))

        # Framenet
        lem2frame = {}
        for lm in framenet.lus():
            for lemma in lm['lexemes']:
                lem2frame[lemma['name'] + '.' + framnet_posdict[lemma['POS']]] = lm['frame']['name']
        frame_names = ['frame=' + x.name for x in framenet.frames()]

        # Verbnet classids
        verbnet_classids = ['classid=' + vcid for vcid in verbnet.classids()]

        dict_feats = {}
        for f in verbnet_classids + lexical_feats + supersenses + frame_names + lcs_feats + all_ud_feature_cols + conc_cols:
            dict_feats[f] = 0

        x_pd = pd.DataFrame([features_func(sent_feat=sent, token=token, lemma=lemma, dict_feats=dict_feats.copy(), prot=args.prot, concreteness=concreteness, lcs=lcs, l2f=lem2frame) for sent, token, lemma in zip(raw_x, tokens, lemmas)])

        dev_x_pd = pd.DataFrame([features_func(sent_feat=sent, token=token, lemma=lemma, dict_feats=dict_feats.copy(), prot=args.prot, concreteness=concreteness, lcs=lcs, l2f=lem2frame) for sent, token, lemma in zip(raw_dev_x, dev_tokens, dev_lemmas)])

        test_x_pd = pd.DataFrame([features_func(sent_feat=sent, token=token, lemma=lemma, dict_feats=dict_feats.copy(), prot=args.prot, concreteness=concreteness, lcs=lcs, l2f=lem2frame) for sent, token, lemma in zip(raw_test_x, test_tokens, test_lemmas)])

        feature_names = (verbnet_classids, supersenses, frame_names, lcs_feats, conc_cols, lexical_feats, all_ud_feature_cols)

        y = {}
        dev_y = {}
        test_y = {}
        for attr in attributes:
            y[attr] = data[attr_map[attr] + ".Norm"].values
            dev_y[attr] = data_dev_mean[attr_map[attr] + ".Norm"].values
            test_y[attr] = data_test_mean[attr_map[attr] + ".Norm"].values

        # get elmo values here
        x_elmo = get_elmo(sentences, tokens=tokens, batch_size=args.batch_size)
        dev_x_elmo = get_elmo(dev_sentences, tokens=dev_tokens, batch_size=args.batch_size)
        test_x_elmo = get_elmo(test_sentences, tokens=test_tokens, batch_size=args.batch_size)

        # Now get glove values
        roots = [sentences[i][tokens[i]] for i in range(len(sentences))]
        dev_roots = [dev_sentences[i][dev_tokens[i]] for i in range(len(dev_sentences))]
        test_roots = [test_sentences[i][test_tokens[i]] for i in range(len(test_sentences))]
        glove_wts = load_glove_embedding(fpath='/srv/models/pytorch/glove/glove.42B.300d', vocab=list(set(roots)), prot=args.prot)
        x_glove = np.array([glove_wts.loc[word] for word in roots])
        dev_x_glove = np.array([glove_wts.loc[word] if word in glove_wts.index else glove_wts.loc["_UNK"]for word in dev_roots])
        test_x_glove = np.array([glove_wts.loc[word] if word in glove_wts.index else glove_wts.loc["_UNK"]for word in test_roots])

        path = '/data/venkat/pickled_data_' + args.modeltype + '/' + args.prot
        with open(path + 'hand.pkl', 'wb') as fout, open(path + 'train_elmo.pkl', 'wb') as train_elmo, open(path + 'dev_elmo.pkl', 'wb') as dev_elmo, open(path + 'test_elmo.pkl', 'wb') as test_elmo, open(path + 'train_glove.pkl', 'wb') as train_glove, open(path + 'dev_glove.pkl', 'wb') as dev_glove, open(path + 'test_glove.pkl', 'wb') as test_glove:
            pickle.dump([x_pd, y, dev_x_pd, dev_y, test_x_pd, test_y, data, data_dev_mean, data_test_mean, feature_names], fout)
            pickle.dump(x_elmo, train_elmo)
            pickle.dump(dev_x_elmo, dev_elmo)
            pickle.dump(test_x_elmo, test_elmo)
            pickle.dump(x_glove, train_glove)
            pickle.dump(dev_x_glove, dev_glove)
            pickle.dump(test_x_glove, test_glove)

        sys.exit("Data has been loaded and pickled")

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
