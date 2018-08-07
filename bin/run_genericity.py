import argparse
import numpy as np
import pandas as pd
from factslab.utility import load_glove_embedding
from sklearn.svm import SVC
from factslab.datastructures import DependencyTree
from math import sqrt
# from allennlp.modules.elmo import Elmo, batch_to_ids

# initialize argument parser
description = 'Run an RNN regression on Genericity protocol annotation.'
parser = argparse.ArgumentParser(description=description)

parser.add_argument('--prot',
                    type=str,
                    default="noun")
parser.add_argument('--structures',
                    type=str,
                    default='structures.tsv')
parser.add_argument('--embeddings',
                    type=str,
                    default='../../../../Downloads/embeddings/glove.42B.300d')

# parse arguments
args = parser.parse_args()

if args.prot == "noun":
    datafile = "noun_data.tsv"
    dev_datafile = "noun_data_dev.tsv"
    response = ["Is.Particular", "Is.Kind", "Is.Abstract"]
    response_conf = ["Part.Confidence", "Kind.Confidence", "Abs.Confidence"]
    attributes = ["part", "kind", "abs"]
    attr_map = {"part": "Is.Particular", "kind": "Is.Kind", "abs": "Is.Abstract"}
    attr_conf = {"part": "Part.Confidence", "kind": "Kind.Confidence",
             "abs": "Abs.Confidence"}
    token_col = "Noun.Token"
else:
    datafile = "pred_data.tsv"
    dev_datafile = "pred_data_dev.tsv"
    response = ["Is.Particular", "Is.Hypothetical", "Is.Dynamic"]
    response_conf = ["Part.Confidence", "Hyp.Confidence", "Dyn.Confidence"]
    attributes = ["part", "hyp", "dyn"]
    attr_map = {"part": "Is.Particular", "dyn": "Is.Dynamic", "hyp": "Is.Hypothetical"}
    attr_conf = {"part": "Part.Confidence", "dyn": "Dyn.Confidence",
             "hyp": "Hyp.Confidence"}
    token_col = "Pred.Root.Token"

data = pd.read_csv(datafile, sep="\t")

data['SentenceID.Token'] = data['Sentence.ID'].map(lambda x: x) + "_" + data[token_col].map(lambda x: str(x))

# Split the datasets into train, dev, test
data_test = data[data['Split'] == 'test']
data = data[data['Split'] != 'test']

# Convert responses to 1s and 0s
for resp in response:
    data[resp] = data[resp].astype(int)

if args.regressiontype == "multinomial":
    # make smallest response value 0
    for resp in response_conf:
        data[resp] = data.groupby('Annotator.ID')[resp].apply(lambda x: x.rank() / (len(x) + 1.))

else:
    # convert response confs to logit ridit scores
    for resp in response_conf:
        data[resp] = data.groupby('Annotator.ID')[resp].apply(lambda x: x.rank() / (len(x) + 1.))
        data[resp] = np.log(data[resp]) - np.log(1. - data[resp])

# Load the structures
structures = {}
vocab = []

# Don't read_csv the structures file. read_csv can't handle quotes
with open(args.structures, 'r') as f:
    for line in f.readlines():
        structs = line.split('\t')
        structures[structs[0]] = DependencyTree.fromstring(structs[1])
        structures[structs[0]].sentence = structs[2].split()
        vocab.append(structs[2].split())
# pdb.set_trace()
data['Structure'] = data['Sentence.ID'].map(lambda x: structures[x])

vocab = list(set(sum(vocab, [])))
data_dev = data[data['Split'] == 'dev']
data = data[data['Split'] == 'train']

# load the glove embedding
embeddings = load_glove_embedding(args.embeddings, vocab)
# ELMO embeddings
# options_file = "/Users/venkat/Downloads/embeddings/options.json"
# weight_file = "/Users/venkat/Downloads/embeddings/weights.hdf5"
# elmo = Elmo(options_file, weight_file, 1, dropout=0.5)

# Shuffle the data
data = data.sample(frac=1).reset_index(drop=True)
data_dev = data_dev.sample(frac=1).reset_index(drop=True)
data_test = data_test.sample(frac=1).reset_index(drop=True)
# Define attributes for regression

raw_x = [struct.sentence for struct in data['Structure']]
tokens = np.array(data[token_col].tolist())

x = []

for i in range(len(raw_x)):
    x.append(embeddings.loc[raw_x[i][tokens[i]]].values)
x = np.array(x)
y = {}
loss_wts = {}

for attr in attributes:
    y[attr] = np.array(data[attr_map[attr]].tolist())
    loss_wts[attr] = np.array(data[attr_conf[attr]].tolist())
    y[attr] = y[attr]

# Initialise dev data
    data_dev = pd.read_csv(dev_datafile, sep="\t")
    data_dev['Structure'] = data_dev['Sentence.ID'].map(lambda x: structures[x])

    raw_dev_x = [struct.sentence for struct in data_dev['Structure']]
    dev_tokens = np.array(data_dev[token_col].tolist())

    dev_x = []
    for i in range(len(raw_dev_x)):
        dev_x.append(embeddings.loc[raw_dev_x[i][dev_tokens[i]]].values)
    dev_x = np.array(dev_x)
    dev_y = {}
    dev_wts = {}
    for attr in attributes:
        dev_y[attr] = data_dev[attr_map[attr]].tolist()
        dev_wts[attr] = data_dev[attr_conf[attr]].tolist()

for attr in attributes:
    # train the model
    trainer = SVC()
    trainer.fit(x, y[attr])

    predictions = trainer.predict(dev_x)

    conf_mat = np.zeros((2, 2))
    for i, output in enumerate(predictions):
        target = dev_y[attr][i]
        conf_mat[int(output)][int(target)] += 1
    print(attr)
    print(conf_mat)
    # Accuracy
    accuracy = sum([conf_mat[i][i] for i in range(2)]) / np.sum(conf_mat)

    # Precision
    p_macro = np.array([conf_mat[i][i] for i in range(2)]) / np.array([sum([conf_mat[j][i] for j in range(2)]) for i in range(2)])
    # Recall
    r_macro = np.array([conf_mat[i][i] for i in range(2)]) / np.array([sum([conf_mat[i][j] for j in range(2)]) for i in range(2)])

    TP = conf_mat[1][1]
    TN = conf_mat[0][0]
    FP = conf_mat[1][0]
    FN = conf_mat[0][1]
    matthews_corr = ((TP * TN) - (FP * FN)) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    # F1 Score
    f1 = np.sum(2 * (p_macro * r_macro) / (p_macro + r_macro)) / 2
    macro_acc = np.sum(p_macro) / 2
    print("Micro Accuracy:", accuracy)
    print("Macro Accuracy:", macro_acc)
    print("Macro F1 score:", f1)
    print("Matthews correlation coefficient:", matthews_corr)
