import argparse
import numpy as np
import pandas as pd
from os.path import expanduser
from torch.nn import LSTM, SmoothL1Loss, L1Loss, CrossEntropyLoss
from factslab.utility import load_glove_embedding
from factslab.datastructures import ConstituencyTree
from factslab.datastructures import DependencyTree
# from factslab.pytorch.childsumtreelstm import ChildSumConstituencyTreeLSTM
from factslab.pytorch.childsumtreelstm import ChildSumDependencyTreeLSTM
from factslab.pytorch.rnnregression import RNNRegressionTrainer
from nltk import DependencyGraph
from random import randint
import pdb
# initialize argument parser
description = 'Run an RNN regression on Genericity protocol annotation.'
parser = argparse.ArgumentParser(description=description)

parser.add_argument('--data',
                    type=str,
                    default='pilot_data_noun.csv')
parser.add_argument('--structures',
                    type=str,
                    default='structures.tsv')
parser.add_argument('--regressiontype',
                    type=str,
                    default="linear")

# parse arguments
args = parser.parse_args()

data = pd.read_csv(args.data)

response = ["part", "kind", "abs"]
response_conf = ["part_conf", "kind_conf", "abs_conf"]

# Convert responses to 1s and 0s
for resp in response:
    data[resp] = data[resp].astype(int)

if args.regressiontype == "multinomial":
    # make smallest response value 0
    for resp in response_conf:
        data[resp] = data[resp].astype(int) - 1         # MAYBE A PROBLEM

else:
    # convert response confs to logit ridit scores
    for resp in response_conf:
        data[resp] = data.groupby('worker_id')[resp].apply(lambda x: x.rank() / (len(x) + 1.))
        data[resp] = np.log(data[resp]) - np.log(1. - data[resp])

# Load the structures
files = ['/UD_English-r1.2/trees-train.tsv',
         '/UD_English-r1.2/trees-dev.tsv']
home = expanduser("~/Downloads/")

structures = []
for file in files:
    path = home + file
    with open(path, 'r') as f:
        structs_sents = [line.strip().split('\t') for line in f]

vocab = []
# Create dependency tree structure and find vocab from sentence
for elem in structs_sents:
    # struct_id.append(elem[0])
    structures.append(DependencyTree.fromstring(elem[1]))
    structures[-1].sentence = elem[2].split()
    vocab.append(elem[2].split())

vocab = list(set(sum(vocab, [])))
# pdb.set_trace()
# load the glove embedding
embeddings = load_glove_embedding('../../../../Downloads/glove.42B.300d',
                                  vocab)


# train the model
trainer = RNNRegressionTrainer(embeddings=embeddings, gpu=False,
                               rnn_classes=ChildSumDependencyTreeLSTM,
                               bidirectional=True, attention=False,
                               regression_type=args.regressiontype,
                               rnn_hidden_sizes=300, num_rnn_layers=1,
                               regression_hidden_sizes=(1,))
# x = [[x.split() for x in list(data['raw_sentence'])]]
x = [[x for x in structures]]
# y = list(data['part'].values)
y = [randint(0, 1) for _ in range(len(structures))]
# y = []
# pdb.set_trace()
trainer.fit(X=x, Y=y, lr=1e-2, batch_size=100, verbosity=1)
