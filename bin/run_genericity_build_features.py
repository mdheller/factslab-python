import argparse
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from predpatt import load_conllu, PredPatt, PredPattOpts
from os.path import expanduser
from collections import Counter, defaultdict
from factslab.utility import ridit, dev_mode_group, get_elmo, load_glove_embedding
import pickle
from nltk.corpus import wordnet, framenet, verbnet


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


def features_func(sent_feat, token, lemma, dict_feats, prot, concreteness, lcs, l2f):
    '''Extract features from a word'''
    sent = sent_feat[0]
    feats = sent_feat[1][0]
    all_lemmas = sent_feat[1][1]
    deps = [x[2] for x in sent.tokens[token].dependents]
    deps_text = [x[2].text for x in sent.tokens[token].dependents]
    deps_feats = '|'.join([(a + "_dep") for x in deps for a in feats[x.position].split('|')])

    all_feats = (feats[token] + '|' + deps_feats).split('|')
    all_feats = list(filter(None, all_feats))

    # UD Lexical features
    for f in all_feats:
        if f in dict_feats.keys():
            dict_feats[f] = 1

    # Lexical item features
    for f in deps_text:
        if f in dict_feats.keys():
            dict_feats[f] = 1

    # wordnet supersense of lemma
    for synset in wordnet.synsets(lemma):
        dict_feats['supersense=' + synset.lexname()] = 1

    # framenet name
    pos = sent.tokens[token].tag
    if lemma + '.' + pos in l2f.keys():
        frame = l2f[lemma + '.' + pos]
        dict_feats['frame=' + frame] = 1

    # Predicate features
    if prot == "pred":
        # verbnet class
        f_lemma = verbnet.classids(lemma=lemma)
        for f in f_lemma:
            dict_feats['classid=' + f] = 1

        # lcs eventiveness
        if lemma in lcs.verbs:
            if True in lcs.eventive(lemma):
                dict_feats['lcs_eventive'] = 1
            else:
                dict_feats['lcs_stative'] = 1

        dep_c_scores = [concreteness_score(concreteness, g_lemma) for g_lemma in [all_lemmas[x[2].position] for x in sent.tokens[token].dependents]]
        if len(dep_c_scores):
            dict_feats['concreteness'] = sum(dep_c_scores) / len(dep_c_scores)
            dict_feats['max_conc'] = max(dep_c_scores)
            dict_feats['min_conc'] = min(dep_c_scores)
        else:
            dict_feats['concreteness'] = 2.5
            dict_feats['max_conc'] = 2.5
            dict_feats['min_conc'] = 2.5
    # Argument features
    else:
        dict_feats['concreteness'] = concreteness_score(concreteness, lemma)

        # lcs eventiveness score and verbnet class of argument head
        if sent.tokens[token].gov:
            gov_lemma = all_lemmas[sent.tokens[token].gov.position]

            # lexical features of dependent of governor
            deps_gov = [x[2].text for x in sent.tokens[token].gov.dependents]
            for f in deps_gov:
                if f in dict_feats.keys():
                    dict_feats[f] = 1

            # lcs eventiveness
            if gov_lemma in lcs.verbs:
                if True in lcs.eventive(gov_lemma):
                    dict_feats['lcs_eventive'] = 1
                else:
                    dict_feats['lcs_stative'] = 1

            for f_lemma in verbnet.classids(lemma=gov_lemma):
                dict_feats['classid=' + f_lemma] = 1

            # framenet name of head
            pos = sent.tokens[token].gov.tag
            if gov_lemma + '.' + pos in l2f.keys():
                frame = l2f[gov_lemma + '.' + pos]
                dict_feats['frame=' + frame] = 1

    return dict_feats


if __name__ == "__main__":
    framnet_posdict = {'V': 'VERB', 'N': 'NOUN', 'A': 'ADJ', 'ADV': 'ADV', 'PREP': 'ADP', 'NUM': 'NUM', 'INTJ': 'INTJ', 'ART': 'DET', 'C': 'CCONJ', 'SCON': 'SCONJ', 'PRON': 'PRON', 'IDIO': 'X', 'AVP': 'ADV'}
        sigdig = 3
        # parse arguments
        args = parser.parse_args()
        home = expanduser('~')

        if args.prot == "arg":
            datafile = home + '/Research/protocols/data/noun_raw_data_norm_122218.tsv'
            attributes = ["part", "kind", "abs"]
            attr_map = {"part": "Is.Particular", "kind": "Is.Kind", "abs": "Is.Abstract"}
            attr_conf = {"part": "Part.Confidence", "kind": "Kind.Confidence",
                     "abs": "Abs.Confidence"}
        else:
            datafile = home + '/Research/protocols/data/pred_raw_data_norm_122218.tsv'
            attributes = ["part", "hyp", "dyn"]
            attr_map = {"part": "Is.Particular", "dyn": "Is.Dynamic", "hyp": "Is.Hypothetical"}
            attr_conf = {"part": "Part.Confidence", "dyn": "Dyn.Confidence",
                     "hyp": "Hyp.Confidence"}

    data = pd.read_csv(datafile, sep="\t")
    data = data.dropna()
    # data['Unique.ID'] = data.apply(lambda x: x['Split'] + " sent_" + str(x['Sentence.ID']) + "_" + str(x["Span"]), axis=1)
    data['Split.Sentence.ID'] = data.apply(lambda x: x['Split'] + " sent_" + x['Sentence.ID'].split('_')[1], axis=1)

    # Load the sentences
    sentences = {}
    with open(home + '/Research/protocols/data/sentences.tsv', 'r') as f:
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
    concreteness = pd.read_csv(home + "/Research/protocols/data/concreteness.tsv", sep="\t")
    if args.prot == 'arg':
        conc_cols = ['concreteness']
    else:
        conc_cols = ['concreteness', 'max_conc', 'min_conc']
    f.close()

    # LCS eventivity
    from lcsreader import LexicalConceptualStructureLexicon
    lcs = LexicalConceptualStructureLexicon(home + '/Research/protocols/data/verbs-English.lcs')
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