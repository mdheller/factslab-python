import os
from os.path import expanduser
import numpy as np
import pandas as pd
from zipfile import ZipFile
from torch import from_numpy, sort
from sklearn.utils import shuffle
from itertools import zip_longest
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, framenet, verbnet
from predpatt import load_conllu, PredPatt, PredPattOpts
import pickle
from collections import Counter


def load_glove_embedding(fpath, vocab):
    """load glove embedding

    Parameters
    ----------
    fpath : str
        path to zip containing glove embeddings
    vocab : list(str)
        list of vocab elements to extract from glove
    """

    zipname = os.path.split(fpath)[-1]
    size, dim = zipname.split('.')[1:3]
    fpathout = 'glove.' + size + '.' + dim + '.filtered.txt'
    if fpathout in os.listdir(os.getcwd()):
        embedding = pd.read_csv(fpathout, index_col=0, header=None, sep=' ')

    else:
        fname = 'glove.' + size + '.' + dim + '.txt'
        f_emb = ZipFile(fpath + '.zip').open(fname)

        emb = np.array([l.decode().strip().split()
                        for l in f_emb
                        if l.split()[0].decode() in vocab])

        embedding = pd.DataFrame(emb[:, 1:].astype(float), index=emb[:, 0])

        mean_emb = list(embedding.mean(axis=0).values)
        oov = [w for w in vocab if w not in embedding.index.values]

        # Add an embedding element for padding
        PADDING_ELEMENT = ["<PAD>"]
        oov = pd.DataFrame(np.tile(mean_emb, [len(oov), 1]), index=oov)
        pad = pd.DataFrame([np.zeros(len(mean_emb))], index=PADDING_ELEMENT)
        oov = pd.concat([oov, pad], axis=0)

        embedding = pd.concat([embedding, oov], axis=0)
        embedding.to_csv(fpathout, sep=' ', header=False)

    return embedding


def load_elmo_embedding(sentences):
    '''
        Takes in list of sentences, batches it and returns embeddings of
        shape num_sentences * longest_sentence_length * (3*1024) 
    '''
    embeddings = 0
    return embeddings


def partition(l, n):
    """partition a list in n blocks"""

    for i in range(0, len(l), n):
        if i < (len(l) - n):
            yield l[i:(i + n)]
        else:
            yield l[i:]


def arrange_inputs(data_batch, targets_batch, wts_batch, tokens_batch, attributes):
        """
            Arrange input sequences so that each minibatch has same length
        """
        sorted_data_batch = []
        sorted_seq_len_batch = []
        sorted_tokens_batch = []
        sorted_idx_batch = []

        sorted_targets_batch = {}
        sorted_wts_batch = {}
        for attr in attributes:
            sorted_targets_batch[attr] = []
            sorted_wts_batch[attr] = []

        for data, tokens in zip(data_batch, tokens_batch):
            seq_len = from_numpy(np.array([len(x) for x in data]))
            sorted_seq_len, sorted_idx = sort(seq_len, descending=True)
            # max_len = sorted_seq_len[0]
            sorted_seq_len_batch.append(np.array(sorted_seq_len))
            sorted_data = [data[x] for x in sorted_idx]
            sorted_tokens = np.array([(tokens[x] + 1) for x in sorted_idx])
            sorted_data_batch.append(sorted_data)
            sorted_tokens_batch.append(sorted_tokens)
            sorted_idx_batch.append(sorted_idx)

        for attr in attributes:
            for targets, wts, sorted_idx in zip(targets_batch[attr], wts_batch[attr], sorted_idx_batch):
                sorted_targets = [targets[x] for x in sorted_idx]
                sorted_wts = [wts[x] for x in sorted_idx]
                sorted_targets_batch[attr].append(sorted_targets)
                sorted_wts_batch[attr].append(sorted_wts)

        return sorted_data_batch, sorted_targets_batch, sorted_wts_batch, sorted_seq_len_batch, sorted_tokens_batch


def ridit(x):
    '''apply ridit scoring

    Parameters
    ----------
    x : iterable

    Returns
    -------
    numpy.array
    '''
    x_flat = np.array(x, dtype=int).flatten()
    x_shift = x_flat - x_flat.min()     # bincount requires nonnegative ints

    bincounts = np.bincount(x_shift)
    props = bincounts / bincounts.sum()

    cumdist = np.cumsum(props)
    cumdist[-1] = 0.                    # this looks odd but is right

    ridit_map = np.array([cumdist[i - 1] + p / 2 for i, p in enumerate(props)])

    return ridit_map[x_shift]


def padding(l1):
    '''
        Adds <PAD> to end of list
    '''
    for minib in l1:
        max_sent_length = max([len(sent) for sent in minib])
        for sent in minib:
            sent += ['<PAD>' for j in range(max_sent_length - len(sent))]
    return l1


def interleave_lists(l1, l2):
    '''
        Interleave two lists and append remaining elements of longer to end
        to form one long list
    '''
    return [item for slist in zip_longest(l1, l2) for item in slist if item is not None]


def dev_mode_group(group, attributes, attr_map, attr_conf):
    '''
        Takes a group from dev data, and returns the (first) mode answer - with mean confidence if all annotations are same, or by changing the conf of non-mode annotations to 1-conf first, then taking the mean of confidences

        Parameters
        ----------
        group
        attributes
        response
        response_conf

        Returns
        -------
        mode_row: pandas Dataframe with just one row
    '''

    mode_row = group.iloc[0]
    for attr in attributes:
        if len(group[attr_map[attr] + ".norm"].unique()) != 1:
            mode_row[attr_map[attr] + ".norm"] = group[attr_map[attr] + ".norm"].mode()[0]
            group[group[attr_map[attr] + ".norm"] != mode_row[attr_map[attr] + ".norm"]][attr_conf[attr] + ".norm"] = 1 - group[group[attr_map[attr] + ".norm"] != mode_row[attr_map[attr] + ".norm"]][attr_conf[attr] + ".norm"]
        mode_row[attr_conf[attr] + ".norm"] = group[attr_conf[attr] + ".norm"].mean()
    return mode_row


def read_data(prot, datafile, attributes, attr_map, attr_conf, regressiontype,
              sentences, batch_size):
    '''
        Reads datafiles, and create minibatched(if desired) lists
        of x, y, tokens, spans, context_roots, context_spans, loss_wts
    '''

    data = pd.read_csv(datafile, sep="\t")
    data = data.dropna()
    data['Split.Sentence.ID'] = data.apply(lambda x: x['Split'] + " sent_" + str(x['Sentence.ID']), axis=1)
    data["Span"] = data["Span"].apply(lambda x: [int(a) for a in x.split(',')])
    if data["Context.Root"].dtype != np.int64:
        data["Context.Root"] = data["Context.Root"].apply(lambda x: [int(a) for a in x.split(',')])
    if 'pred' in datafile:
        data["Context.Span"] = data["Context.Span"].apply(lambda x: [list(map(int, b.split(','))) for b in x.split(';')])
    else:
        data["Context.Span"] = data["Context.Span"].apply(lambda x: [int(a) for a in x.split(',')])
    data['Unique.ID'] = data.apply(lambda x: x['Split'] + " sent_" + str(x['Sentence.ID']) + "_" + str(x["Span"]), axis=1)
    data['Sentence'] = data['Split.Sentence.ID'].map(lambda x: sentences[x])

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
        if regressiontype == "multinomial":
            data[resp + ".norm"] = data[resp].map(lambda x: 1 if x else 0)
            data_dev[resp + ".norm"] = data_dev[resp].map(lambda x: 1 if x else 0)
        elif regressiontype == "linear":
            data[resp + ".norm"] = data[resp].map(lambda x: 1 if x else -1) * data[resp_conf + ".norm"]
            data_dev[resp + ".norm"] = data_dev[resp].map(lambda x: 1 if x else -1) * data_dev[resp_conf + ".norm"]

    # Shuffle the data
    data = shuffle(data).reset_index(drop=True)
    data_dev = shuffle(data_dev).reset_index(drop=True)
    data_test = shuffle(data_test).reset_index(drop=True)

    # Prepare all the inputs for the neural model
    x = [[datum[:] for datum in data['Sentence'].values.tolist()][i:i + batch_size] for i in range(0, len(data['Sentence']), batch_size)]

    roots = [data["Root.Token"].values.tolist()[i:i + batch_size] for i in range(0, len(data["Root.Token"]), batch_size)]

    context_roots = [data["Context.Root"].values.tolist()[i:i + batch_size] for i in range(0, len(data["Context.Root"]), batch_size)]

    context_spans = [[datum[:] for datum in data['Context.Span'].values.tolist()][i:i + batch_size] for i in range(0, len(data["Context.Span"]), batch_size)]

    # Form tuples from the contexts
    spans = [[datum[:] for datum in data['Span'].values.tolist()][i:i + batch_size] for i in range(0, len(data["Span"]), batch_size)]

    y = [{attr: (data[attr_map[attr] + ".norm"].values[i:i + batch_size]) for attr in attributes} for i in range(0, len(data[attr_map[attr] + ".norm"].values), batch_size)]

    loss_wts = [{attr: data[attr_conf[attr] + ".norm"].values[i:i + batch_size] for attr in attributes} for i in range(0, len(data[attr_conf[attr] + ".norm"].values), batch_size)]

    # Create dev data
    if regressiontype == "linear":
        data_dev_mean = data_dev.groupby('Unique.ID', as_index=False).mean()
    else:
        data_dev_mean = data_dev.groupby('Unique.ID', as_index=False).apply(lambda x: dev_mode_group(x, attributes, attr_map, attr_conf)).reset_index(drop=True)

    data_dev_mean['Sentence'] = data_dev_mean['Unique.ID'].map(lambda x: data_dev[data_dev['Unique.ID'] == x]['Sentence'].iloc[0])

    dev_x = [[datum[:] for datum in data_dev_mean['Sentence'].values.tolist()][i:i + batch_size] for i in range(0, len(data_dev_mean['Sentence']), batch_size)]

    dev_roots = [data_dev_mean["Root.Token"].values.tolist()[i:i + batch_size] for i in range(0, len(data_dev_mean["Root.Token"]), batch_size)]

    dev_context_roots = [data_dev_mean["Context.Root"].values.tolist()[i:i + batch_size] for i in range(0, len(data_dev_mean["Context.Root"]), batch_size)]

    dev_context_spans = [[datum[:] for datum in data_dev_mean['Context.Span'].values.tolist()][i:i + batch_size] for i in range(0, len(data_dev_mean["Context.Span"]), batch_size)]

    dev_spans = [[datum[:] for datum in data_dev_mean['Span'].values.tolist()][i:i + batch_size] for i in range(0, len(data_dev_mean["Span"]), batch_size)]

    dev_y = {}
    dev_wts = {}
    for attr in attributes:
        dev_y[attr] = data_dev_mean[attr_map[attr] + ".norm"].values
        dev_wts[attr] = data_dev_mean[attr_conf[attr] + ".norm"].values

    # Prepare hand engineered features
    hand_feats, hand_feats_dev = hand_engineering(batch_size=batch_size,
                                                  prot=prot, data=data,
                                                  data_dev=data_dev_mean)

    dev = [dev_x, dev_y, dev_roots, dev_spans, dev_context_roots, dev_context_spans, dev_wts, hand_feats_dev]
    train = [x, y, roots, spans, context_roots, context_spans, loss_wts, hand_feats]

    return (train, dev)


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
    '''
        Extract hand engineered features from a word
    '''
    lmt = WordNetLemmatizer()
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


def hand_engineering(prot, batch_size, data, data_dev):
    '''
        Hand engineered feature extraction. Supports the following - UD,
        Verbnet classids, Wordnet supersenses, concreteness ratings, LCS
        eventivity scores
    '''
    home = expanduser("~")
    # Load the features
    features = {}
    with open(home + '/Desktop/protocols/data/features.tsv', 'r') as f:
        for line in f.readlines():
            feats = line.split('\t')
            features[feats[0]] = feats[1].split()

    # Load the predpatt objects for creating features
    files = ['/Downloads/UD_English-r1.2/en-ud-train.conllu',
             '/Downloads/UD_English-r1.2/en-ud-dev.conllu',
             '/Downloads/UD_English-r1.2/en-ud-test.conllu']
    home = expanduser("~")
    options = PredPattOpts(resolve_relcl=True, borrow_arg_for_relcl=True, resolve_conj=False, cut=True)  # Resolve relative clause
    patt = {}

    for file in files:
        path = home + file
        with open(path, 'r') as infile:
            for sent_id, ud_parse in load_conllu(infile.read()):
                patt[file[33:][:-7] + " " + sent_id] = PredPatt(ud_parse, opts=options)

    data['Structure'] = data['Split.Sentence.ID'].map(lambda x: (patt[x], features[x]))
    data_dev['Structure'] = data_dev['Split.Sentence.ID'].map(lambda x: (patt[x], features[x]))

    raw_x = data['Structure'].tolist()
    raw_dev_x = data_dev['Structure'].tolist()

    all_x = raw_x + raw_dev_x
    all_feats = '|'.join(['|'.join(all_x[i][1]) for i in range(len(all_x))])
    feature_cols = Counter(all_feats.split('|'))
    dict_feats = {}

    f = open(home + '/Desktop/protocols/data/concrete.pkl', 'rb')
    concreteness = pickle.load(f)
    f.close()

    from factslab.utility.lcsreader import LexicalConceptualStructureLexicon
    lcs = LexicalConceptualStructureLexicon(home + '/Desktop/protocols/data/verbs-English.lcs')

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

    x_pd = pd.DataFrame([features_func(sent_feat=sent, token=token, lemma=lemma, dict_feats=dict_feats.copy(), prot=prot, concreteness=concreteness, lcs=lcs) for sent, token, lemma in zip(raw_x, data['Root.Token'].tolist(), data['Lemma'].tolist())])

    dev_x_pd = pd.DataFrame([features_func(sent_feat=sent, token=token, lemma=lemma, dict_feats=dict_feats.copy(), prot=prot, concreteness=concreteness, lcs=lcs) for sent, token, lemma in zip(raw_dev_x, data_dev['Root.Token'].tolist(), data_dev['Lemma'].tolist())])

    # Figure out which columns to drop(they're always zero)
    todrop1 = dev_x_pd.columns[(dev_x_pd == 0).all()].values.tolist()
    todrop = x_pd.columns[(x_pd == 0).all()].values.tolist()
    intdrop = [a for a in todrop if a not in todrop1]
    cols_to_drop = cols_to_drop = list(set(todrop) - set(intdrop))

    x = x_pd.drop(cols_to_drop, axis=1).values.tolist()
    dev_x = dev_x_pd.drop(cols_to_drop, axis=1).values.tolist()

    x = [[a[:] for a in x[i:i + batch_size]] for i in range(0, len(data), batch_size)]
    dev_x = [[a[:] for a in dev_x[i:i + batch_size]] for i in range(0, len(data_dev), batch_size)]
    return x, dev_x


def feature_extract():
    '''
        Extract all the UD lexical features and write to a file
    '''
    from os.path import expanduser

    files = ['Downloads/UD_English-r1.2/en-ud-train.conllu',
             'Downloads/UD_English-r1.2/en-ud-dev.conllu',
             'Downloads/UD_English-r1.2/en-ud-test.conllu']
    home = expanduser("~")

    for file in files:
        with open(home + '/Desktop/protocols/data/features.tsv', 'a') as fout:
            path = home + file
            with open(path, 'r') as f:
                id = 0
                feats = []
                for line in f:
                    if line != "\n":
                        all_feats = line.split("\t")
                        feats.append("UPOS=" + all_feats[3] + "|" + "XPOS=" + all_feats[4] + "|" + all_feats[5] + "|" + "DEPREL=" + all_feats[7])
                    else:
                        id += 1
                        sent_id = file[23:][:-7] + " sent_" + str(id)
                        feats = " ".join(feats)
                        fout.write(sent_id + "\t" + feats + "\n")
                        feats = []
