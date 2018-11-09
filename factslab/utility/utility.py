import os
import numpy as np
import pandas as pd
from zipfile import ZipFile
from torch import from_numpy, sort
from sklearn.utils import shuffle
from itertools import zip_longest


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


def read_data(datafile, attributes, attr_map, attr_conf, regressiontype,
              structures, batch_size):
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
    data['Structure'] = data['Split.Sentence.ID'].map(lambda x: structures[x])

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
    x = [[datum[:] for datum in data['Structure'].values.tolist()][i:i + batch_size] for i in range(0, len(data['Structure']), batch_size)]

    tokens = [data["Root.Token"].values.tolist()[i:i + batch_size] for i in range(0, len(data["Root.Token"]), batch_size)]

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

    data_dev_mean['Structure'] = data_dev_mean['Unique.ID'].map(lambda x: data_dev[data_dev['Unique.ID'] == x]['Structure'].iloc[0])

    dev_x = [[datum[:] for datum in data_dev_mean['Structure'].values.tolist()][i:i + batch_size] for i in range(0, len(data_dev_mean['Structure']), batch_size)]

    dev_tokens = [data_dev_mean["Root.Token"].values.tolist()[i:i + batch_size] for i in range(0, len(data_dev_mean["Root.Token"]), batch_size)]

    dev_context_roots = [data_dev_mean["Context.Root"].values.tolist()[i:i + batch_size] for i in range(0, len(data_dev_mean["Context.Root"]), batch_size)]

    dev_context_spans = [[datum[:] for datum in data_dev_mean['Context.Span'].values.tolist()][i:i + batch_size] for i in range(0, len(data_dev_mean["Context.Span"]), batch_size)]

    dev_spans = [[datum[:] for datum in data_dev_mean['Span'].values.tolist()][i:i + batch_size] for i in range(0, len(data_dev_mean["Span"]), batch_size)]

    dev_y = {}
    dev_wts = {}
    for attr in attributes:
        dev_y[attr] = data_dev_mean[attr_map[attr] + ".norm"].values
        dev_wts[attr] = data_dev_mean[attr_conf[attr] + ".norm"].values

    dev = [dev_x, dev_y, dev_tokens, dev_spans, dev_context_roots, dev_context_spans, dev_wts]
    train = [x, y, tokens, spans, context_roots, context_spans, loss_wts]

    return (train, dev)
