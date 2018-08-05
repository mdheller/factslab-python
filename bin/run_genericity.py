import argparse
import numpy as np
import pandas as pd
from factslab.utility import load_glove_embedding, arrange_inputs
from torch.nn import LSTM
# from factslab.datastructures import ConstituencyTree
from factslab.datastructures import DependencyTree
from factslab.pytorch.childsumtreelstm import ChildSumDependencyTreeLSTM
from factslab.pytorch.rnnregression import RNNRegressionTrainer
from torch.cuda import is_available
from torch import device
from torch import max
import sys
from sklearn.svm import SVC

# initialize argument parser
description = 'Run an RNN regression on Genericity protocol annotation.'
parser = argparse.ArgumentParser(description=description)

parser.add_argument('--attr',
                    type=str,
                    default="noun")
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
parser.add_argument('--lr',
                    type=float,
                    default="0.01")
parser.add_argument('--L2',
                    type=float,
                    default="0")
parser.add_argument('--attention',
                    action='store_true',
                    help='Turn attention on or off')

# parse arguments
args = parser.parse_args()

if args.attr == "noun":
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
# For elmo pre-trained embeddings
# embeddings = h5py.File('../../../../Downloads/embeddings/embeddings.hdf5', 'r')

# pyTorch figures out device to do computation on
device_to_use = device("cuda:0" if is_available() else "cpu")
# Shuffle the data
data = data.sample(frac=1).reset_index(drop=True)
data_dev = data_dev.sample(frac=1).reset_index(drop=True)
data_test = data_test.sample(frac=1).reset_index(drop=True)
# Define attributes for regression

if args.rnntype == "tree":
    # Handle the tree input structuring here. No minibatch
    rnntype = ChildSumDependencyTreeLSTM
    x_raw = [struct for struct in data['Structure']]
    x = [x_raw[i:i + args.batch] for i in range(0, len(x_raw), args.batch)]
    y_raw = data[attr].values
    y = [y_raw[i:i + args.batch] for i in range(0, len(y_raw), args.batch)]
    wt_raw = data[attr_conf].values
    loss_wts = [wt_raw[i:i + args.batch] for i in range(0, len(wt_raw), args.batch)]

elif args.rnntype == "linear":
    # Handle linear LSTM here. Minibatch can be done
    rnntype = LSTM
    x_raw = [struct.sentence for struct in data['Structure']]
    x = [x_raw[i:i + args.batch] for i in range(0, len(x_raw), args.batch)]
    tokens_raw = data[token_col].values
    tokens = [tokens_raw[i:i + args.batch] for i in range(0, len(tokens_raw), args.batch)]
    y = {}
    loss_wts = {}
    for attr in attributes:
        y_raw = data[attr_map[attr]].values
        y[attr] = [y_raw[i:i + args.batch] for i in range(0, len(y_raw), args.batch)]
        y[attr][-1] = np.append(y[attr][-1], y[attr][-2][0:len(y[attr][-2]) - len(y[attr][-1])])
        wt_raw = data[attr_conf[attr]].values
        loss_wts[attr] = [wt_raw[i:i + args.batch] for i in range(0, len(wt_raw), args.batch)]
        loss_wts[attr][-1] = np.append(loss_wts[attr][-1], loss_wts[attr][-2][0:len(loss_wts[attr][-2]) - len(loss_wts[attr][-1])])

    x[-1] = x[-1] + x[-2][0:len(x[-2]) - len(x[-1])]    # last batch size hack
    tokens[-1] = np.append(tokens[-1], tokens[-2][0:len(tokens[-2]) - len(tokens[-1])])

    # Arrange in descending order and pad each batch
    x, y, loss_wts, lengths, tokens = arrange_inputs(data_batch=x,
                                                     targets_batch=y,
                                                     wts_batch=loss_wts,
                                                     tokens_batch=tokens,
                                                     attributes=attributes)
else:
    sys.exit('Error. Argument rnntype must be tree or linear')

# Initialise dev data
data_dev = pd.read_csv(dev_datafile, sep="\t")
data_dev['Structure'] = data_dev['Sentence.ID'].map(lambda x: structures[x])

dev_x = [struct.sentence for struct in data_dev['Structure']]
dev_tokens = np.array(data_dev[token_col].tolist()) + 1

dev_y = {}
dev_wts = {}
for attr in attributes:
    dev_y[attr] = data_dev[attr_map[attr]].tolist()
    dev_wts[attr] = data_dev[attr_conf[attr]].tolist()

# train the model
trainer = RNNRegressionTrainer(embeddings=embeddings, device=device_to_use,
                               rnn_classes=rnntype,
                               bidirectional=True, attention=False,
                               regression_type=args.regressiontype,
                               rnn_hidden_sizes=300, num_rnn_layers=1,
                               regression_hidden_sizes=(150,),
                               epochs=args.epochs, batch_size=args.batch,
                               attributes=attributes)

trainer.fit(X=x, Y=y, tokens=tokens, verbosity=args.verbosity, loss_weights=loss_wts, lengths=lengths, lr=args.lr, weight_decay=args.L2, dev=[dev_x, dev_tokens, dev_y, dev_wts])
