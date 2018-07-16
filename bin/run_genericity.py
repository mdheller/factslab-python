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
from torch import max
import pdb
import sys
from torch import from_numpy, sort
from torch import save
from torch import load
from statistics import mode
# import h5py


def _arrange_inputs(data_batch, targets_batch, wts_batch, tokens_batch):
        """
            Arrange input sequences so that each minibatch has same length
        """
        sorted_data_batch = []
        sorted_targets_batch = []
        sorted_seq_len_batch = []
        sorted_wts_batch = []
        sorted_tokens_batch = []
        for data, targets, wts, tokens in zip(data_batch, targets_batch, wts_batch, tokens_batch):
            seq_len = from_numpy(np.array([len(x) for x in data]))
            sorted_seq_len, sorted_idx = sort(seq_len, descending=True)
            max_len = sorted_seq_len[0]
            sorted_seq_len_batch.append(np.array(sorted_seq_len))
            sorted_data = [data[x] + ['<PAD>' for i in range(max_len - len(data[x]))] for x in sorted_idx]
            sorted_targets = [targets[x] for x in sorted_idx]
            sorted_wts = [wts[x] for x in sorted_idx]
            sorted_tokens = np.array([(tokens[x] + 1) for x in sorted_idx])
            sorted_data_batch.append(sorted_data)
            sorted_targets_batch.append(sorted_targets)
            sorted_wts_batch.append(sorted_wts)
            sorted_tokens_batch.append(sorted_tokens)

        return sorted_data_batch, sorted_targets_batch, sorted_wts_batch, sorted_seq_len_batch, sorted_tokens_batch


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
                    default="10")
parser.add_argument('--attribute',
                    type=str,
                    default="noun_part")

# parse arguments
args = parser.parse_args()

# Choose the attribute based on arguments
attribute_map = {
                 "noun_part": ("Is.Particular", "Part.Confidence", "Noun.Token"),
                 "noun_abs": ("Is.Abstract", "Abs.Confidence", "Noun.Token"),
                 "noun_kind": ("Is.Kind", "Kind.Confidence", "Noun.Token"),
                 "pred_part": ("Is.Particular", "Part.Confidence", "Pred.Token"),
                 "pred_hyp": ("Is.Hypothetical", "Hyp.Confidence", "Pred.Token"),
                 "pred_dyn": ("Is.Dynamic", "Dyn.Confidence", "Pred.Token")
                }

attr = attribute_map[args.attribute][0]
attr_conf = attribute_map[args.attribute][1]
attr_token = attribute_map[args.attribute][2]
data = pd.read_csv(args.data, sep="\t")

data['SentenceID.Token'] = data['Sentence.ID'].map(lambda x: x) + "_" + data['Noun.Token'].map(lambda x: str(x))
response = ["Is.Particular", "Is.Kind", "Is.Abstract"]
response_conf = ["Part.Confidence", "Kind.Confidence", "Abs.Confidence"]

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
data_test = data[data['Split'] == 'test']
data = data[data['Split'] == 'train']

# load the glove embedding
embeddings = load_glove_embedding(args.embeddings, vocab)
# For elmo pre-trained embeddings
# embeddings = h5py.File('../../../../Downloads/embeddings/embeddings.hdf5', 'r')

# pyTorch figures out device to do computation on
device_to_use = device("cuda:0" if is_available() else "cpu")
data = data.sample(frac=1).reset_index(drop=True)
data_dev = data_dev.sample(frac=1).reset_index(drop=True)
data_test = data_test.sample(frac=1).reset_index(drop=True)

if args.rnntype == "tree":
    x_raw = [struct for struct in data['Structure']]
    x = [x_raw[i:i + args.batch] for i in range(0, len(x_raw), args.batch)]
    y_raw = data[attr].values
    y = [y_raw[i:i + args.batch] for i in range(0, len(y_raw), args.batch)]
    wt_raw = data[attr_conf].values
    loss_weights = [wt_raw[i:i + args.batch] for i in range(0, len(wt_raw), args.batch)]
    rnntype = ChildSumDependencyTreeLSTM
    args.batch = 1
elif args.rnntype == "linear":
    # Implmenent mini-batching
    x_raw = [struct.sentence for struct in data['Structure']]
    x = [x_raw[i:i + args.batch] for i in range(0, len(x_raw), args.batch)]
    y_raw = data[attr].values
    y = [y_raw[i:i + args.batch] for i in range(0, len(y_raw), args.batch)]
    wt_raw = data[attr_conf].values
    loss_weights = [wt_raw[i:i + args.batch] for i in range(0, len(wt_raw), args.batch)]
    tokens_raw = data["Noun.Token"].values
    tokens = [tokens_raw[i:i + args.batch] for i in range(0, len(tokens_raw), args.batch)]
    rnntype = LSTM
    # Take care of the fact that the last batch doesn't have size 128. This
    # casues problems in attention
    x[-1] = x[-1] + x[-2][0:len(x[-2]) - len(x[-1])]
    y[-1] = np.append(y[-1], y[-2][0:len(y[-2]) - len(y[-1])])
    tokens[-1] = np.append(tokens[-1], tokens[-2][0:len(tokens[-2]) - len(tokens[-1])])
    loss_weights[-1] = np.append(loss_weights[-1], loss_weights[-2][0:len(loss_weights[-2]) - len(loss_weights[-1])])
    # Arrange in descending order and pad each batch
    x, y, loss_weights, lengths, tokens = _arrange_inputs(data_batch=x,
                                              targets_batch=y,
                                              wts_batch=loss_weights,
                                              tokens_batch=tokens)
else:
    sys.exit('Error. Argument rnntype must be tree or linear')

# train the model
trainer = RNNRegressionTrainer(embeddings=embeddings, device=device_to_use,
                               rnn_classes=rnntype,
                               bidirectional=True, attention=False,
                               regression_type=args.regressiontype,
                               rnn_hidden_sizes=300, num_rnn_layers=1,
                               regression_hidden_sizes=(150,),
                               epochs=args.epochs, batch_size=args.batch)

trainer.fit(X=x, Y=y, lr=1e-2, tokens=tokens, verbosity=args.verbosity, loss_weights=loss_weights, lengths=lengths)
# trainer._regression.load_state_dict(load('trainer.dat'))

# Now to do prediction on developement dataset

# Choose the mode answer and apply 1-conf to minority answer confidence
# sent_ids = list(set(data_dev['SentenceID.Token'].tolist()))
# data_dev_reduced = pd.DataFrame()
# for sent_id in sent_ids:
#     new_df = data_dev[data_dev['SentenceID.Token'] == sent_id]
#     sample = new_df.iloc[0]

#     answers = new_df[attr].tolist()
#     if all(x == answers[0] for x in answers):
#         mode_ans = answers[0]
#         new_conf = sum(new_df[attr_conf].tolist()) / 3
#     else:
#         mode_ans = mode(answers)
#         new_df[new_df[attr] != mode_ans][attr_conf] = 1 - new_df[new_df[attr] != mode_ans][attr_conf]
#         new_conf = sum(new_df[attr_conf].tolist()) / 3

#     sample[attr] = mode_ans
#     sample[attr_conf] = new_conf
#     data_dev_reduced = data_dev_reduced.append(sample)

dev_x = [struct.sentence for struct in data_dev['Structure']]
dev_y = data_dev[attr].tolist()
dev_wts = data_dev[attr_conf].tolist()
dev_tokens = np.array(data_dev["Noun.Token"].tolist()) + 1
Ns = 2
conf_mat = np.zeros((Ns, Ns))
wt_conf_mat = np.zeros((Ns, Ns))
outputs = trainer.predict(X=dev_x, tokens=dev_tokens)
for output, target, wt in zip(outputs, dev_y, dev_wts):
    _, ind = max(output, 0)
    conf_mat[int(ind)][int(target)] += 1
    wt_conf_mat[int(ind)][int(target)] += wt * 1

# Accuracy
accuracy = sum([conf_mat[i][i] for i in range(Ns)]) / np.sum(conf_mat)

# Precision
p_macro = np.array([conf_mat[i][i] for i in range(Ns)]) / np.array([sum([conf_mat[j][i] for j in range(Ns)]) for i in range(Ns)])
# Recall
r_macro = np.array([conf_mat[i][i] for i in range(Ns)]) / np.array([sum([conf_mat[i][j] for j in range(Ns)]) for i in range(Ns)])

# F1 Score
f1 = np.sum(2 * (p_macro * r_macro) / (p_macro + r_macro)) / Ns
macro_acc = np.sum(p_macro) / Ns
print("Micro Accuracy:", accuracy)
print("Macro Accuracy:", macro_acc)
print("Macro F1 score:", f1)
