import argparse
import numpy as np
import pandas as pd
from os.path import expanduser
from torch.nn import LSTM, SmoothL1Loss, L1Loss, CrossEntropyLoss
from factslab.utility import load_glove_embedding
from factslab.datastructures import ConstituencyTree
from factslab.datastructures import DependencyTree
from factslab.pytorch.childsumtreelstm import ChildSumDependencyTreeLSTM
from factslab.pytorch.rnnregression import RNNRegressionTrainer
from nltk import DependencyGraph
from random import randint
from torch.cuda import is_available
from torch import device
import pdb
import sys
# import h5py

# initialize argument parser
description = 'Run an RNN regression on Genericity protocol annotation.'
parser = argparse.ArgumentParser(description=description)

parser.add_argument('--data',
                    type=str,
                    default='noun_data.tsv')
parser.add_argument('--structures',
                    type=str,
                    default='structures.tsv')
parser.add_argument('--embeddings',
                    type=str,
                    default='../../../../Downloads/embeddings/glove.42B.300d')
parser.add_argument('--regressiontype',
                    type=str,
                    default="linear")
parser.add_argument('--epochs',
                    type=int,
                    default=1)
parser.add_argument('--batch',
                    type=int,
                    default=128)
parser.add_argument('--rnntype',
                    type=str,
                    default="tree")
parser.add_argument('--verbosity',
                    type=int,
                    default="1")

# parse arguments
args = parser.parse_args()

data = pd.read_csv(args.data, sep="\t")

data = data[data['Split'] == 'train']
dev_data = data[data['Split'] == 'dev']
test_data = data[data['Split'] == 'test']

data['SentenceID.Token'] = data['Sentence.ID'].map(lambda x: x) + "_" + data['Noun.Token'].map(lambda x: str(x))
response = ["Is.Particular", "Is.Kind", "Is.Abstract"]
response_conf = ["Part.Confidence", "Kind.Confidence", "Abs.Confidence"]

# Convert responses to 1s and 0s
for resp in response:
    data[resp] = data[resp].astype(int)

if args.regressiontype == "multinomial":
    # make smallest response value 0
    for resp in response_conf:
        data[resp] = data[resp].astype(int)

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
# data.map(lambda x: x['Structure'].token=x['Noun.Token'])
# data['Structure'] = structures[data['SentenceID.Token']]

vocab = list(set(sum(vocab, [])))


# load the glove embedding
embeddings = load_glove_embedding(args.embeddings, vocab)
# For elmo pre-trained embeddings
# embeddings = h5py.File('../../../../Downloads/embeddings/embeddings.hdf5', 'r')

# pyTorch figures out device to do computation on
device_to_use = device("cuda:0" if is_available() else "cpu")
data.reindex(np.random.permutation(data.index))
if args.rnntype == "tree":
    x_raw = [struct for struct in data['Structure']]
    y_raw = data["Is.Particular"].values
    rnntype = ChildSumDependencyTreeLSTM
elif args.rnntype == "linear":
    # Implmenent mini-batching
    x_raw = [struct.sentence for struct in data['Structure']]
    y_raw = data["Is.Particular"].values
    rnntype = LSTM
else:
    sys.exit('Error. Argument rnntype must be tree or linear')

x = [x_raw[i:i + args.batch] for i in range(0, len(x_raw), args.batch)]
y = [y_raw[i:i + args.batch] for i in range(0, len(y_raw), args.batch)]
wt_raw = data["Part.Confidence"].values
#wt_raw = []
loss_weights = [wt_raw[i:i + args.batch] for i in range(0, len(wt_raw), args.batch)]

# train the model
trainer = RNNRegressionTrainer(embeddings=embeddings, device=device_to_use,
                               rnn_classes=rnntype, attributes=["part"],
                               bidirectional=True, attention=False,
                               regression_type=args.regressiontype,
                               rnn_hidden_sizes=300, num_rnn_layers=1,
                               regression_hidden_sizes=(150,),
                               epochs=args.epochs)

trainer.fit(X=x, Y=y, lr=1e-2, batch_size=args.batch, verbosity=args.verbosity,
            loss_weights=loss_weights)

# Now to do prediction on test