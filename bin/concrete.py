import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import pickle
from os.path import expanduser


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


if __name__ == "__main__":
    home = expanduser('~')
    datafile = home + '/Desktop/protocols/data/arg_long_data.tsv'
    response = ["Is.Particular", "Is.Kind", "Is.Abstract"]
    response_conf = ["Part.Confidence", "Kind.Confidence", "Abs.Confidence"]
    attributes = ["part", "kind", "abs"]
    attr_map = {"part": "Is.Particular", "kind": "Is.Kind", "abs": "Is.Abstract"}
    attr_conf = {"part": "Part.Confidence", "kind": "Kind.Confidence",
             "abs": "Abs.Confidence"}
    token_col = "Noun.Token"

    data = pd.read_csv(datafile, sep="\t")

    # data['SentenceID.Token'] = data['Sentence.ID'].map(lambda x: x) + "_" + data[token_col].map(lambda x: str(x))

    # Split the datasets into train, dev, test
    # data_test = data[data['Split'] == 'test']
    # data = data[data['Split'] != 'test']
    # data_dev = data[data['Split'] != 'dev']

    # Ridit scoring annotations and confidence ratings
    for attr in attributes:
        resp = attr_map[attr]
        resp_conf = attr_conf[attr]
        data['ridit_' + resp_conf] = data.groupby('Annotator.ID')[resp_conf].transform(ridit)
        data[resp + ".norm"] = data[resp].map(lambda x: 1 if x else -1) * data['ridit_' + resp_conf]

    path = "concreteness.tsv"
    concreteness = pd.read_csv(path, sep="\t")
    list_of_lemmas = concreteness['Word'].values.tolist()

    with open('concrete.pkl', 'wb') as f:
        pickle.dump(concreteness, f)
    abs_conc = data.groupby('Lemma')['Is.Abstract.norm'].median().to_frame().reset_index()
    abs_conc['Concreteness'] = abs_conc['Lemma'].map(lambda x: concreteness[concreteness['Word'] == x.lower()]['Conc.M'].values[0] if x.lower() in list_of_lemmas else -1)
    ini = len(abs_conc)
    abs_conc = abs_conc[abs_conc['Concreteness'] != -1]
    print(len(abs_conc) / ini)
    print("Spearman correlation: ", spearmanr(abs_conc['Is.Abstract.norm'].values.tolist(), abs_conc['Concreteness'].values.tolist())[0])
    print("Pearson correlation: ", np.corrcoef(abs_conc['Is.Abstract.norm'].values.tolist(), abs_conc['Concreteness'].values.tolist())[0][1])
