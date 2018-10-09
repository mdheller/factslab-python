import argparse
import numpy as np
import pandas as pd
from factslab.utility import ridit
from factslab.pytorch.mlpregression import MLPTrainer
from torch.cuda import is_available
from torch import device
from allennlp.modules.elmo import Elmo
from sklearn.utils import shuffle
from os.path import expanduser

home = expanduser('~')
# initialize argument parser
description = 'Run a simple MLP with(out) attention on varying levels on ELMO.'
parser = argparse.ArgumentParser(description=description)

parser.add_argument('--protocol',
                    type=str,
                    default="noun")
parser.add_argument('--structures',
                    type=str,
                    default=home + '/Desktop/protocols/data/structures.tsv')
parser.add_argument('--embeddings',
                    type=str,
                    default=home + '/Downloads/embeddings/')
parser.add_argument('--regressiontype',
                    type=str,
                    default="linear")
parser.add_argument('--epochs',
                    type=int,
                    default=1)
parser.add_argument('--batch',
                    type=int,
                    default=128)
parser.add_argument('--verbosity',
                    type=int,
                    default="1")
parser.add_argument('--attention',
                    action='store_true',
                    help='Turn attention on or off')
parser.add_argument('--span',
                    action='store_true',
                    help='Turn span attention on or off')
parser.add_argument('--sentence',
                    action='store_true',
                    help='Turn sentence attention on or off')
parser.add_argument('--paramattention',
                    action='store_true',
                    help='Turn param attention on or off')

# parse arguments
args = parser.parse_args()

# Find out the attention type to be used based on arguments
if not args.attention:
    attention_type = "None"
else:
    if not args.paramattention:
        if args.span and not args.sentence:
            attention_type = "Span"
        elif args.sentence and not args.span:
            attention_type = "Sentence"
        else:
            # Span + sentence?
            pass
    else:
        if args.span and not args.sentence:
            attention_type = "Span-param"
        elif args.sentence and not args.span:
            attention_type = "Sentence-param"
        else:
            # Span + sentence?
            pass

if args.protocol == "noun":
    datafile = "../../../protocols/data/noun_long_data.tsv"
    response = ["Is.Particular", "Is.Kind", "Is.Abstract"]
    response_conf = ["Part.Confidence", "Kind.Confidence", "Abs.Confidence"]
    attributes = ["part", "kind", "abs"]
    attr_map = {"part": "Is.Particular", "kind": "Is.Kind", "abs": "Is.Abstract"}
    attr_conf = {"part": "Part.Confidence", "kind": "Kind.Confidence",
             "abs": "Abs.Confidence"}
    token_col = "Noun.Token"
    root_token = "Noun.Root.Token"
else:
    datafile = "../../../protocols/data/pred_long_data.tsv"
    response = ["Is.Particular", "Is.Hypothetical", "Is.Dynamic"]
    response_conf = ["Part.Confidence", "Hyp.Confidence", "Dyn.Confidence"]
    attributes = ["part", "hyp", "dyn"]
    attr_map = {"part": "Is.Particular", "dyn": "Is.Dynamic", "hyp": "Is.Hypothetical"}
    attr_conf = {"part": "Part.Confidence", "dyn": "Dyn.Confidence",
             "hyp": "Hyp.Confidence"}
    token_col = "Pred.Token"
    root_token = "Pred.Root.Token"

data = pd.read_csv(datafile, sep="\t")

data['Split.Sentence.ID'] = data.apply(lambda x: x['Split'] + " sent_" + str(x['Sentence.ID']), axis=1)

data['Unique.ID'] = data.apply(lambda x: x['Split'] + " sent_" + str(x['Sentence.ID']) + "_" + str(x[token_col]), axis=1)

# Load the structures
structures = {}

# Don't read_csv the structures file. read_csv can't handle quotes
with open(args.structures, 'r') as f:
    for line in f.readlines():
        structs = line.split('\t')
        structures[structs[0]] = structs[1].split()

data['Structure'] = data['Split.Sentence.ID'].map(lambda x: structures[x])

# Split the datasets into train, dev, test
data_test = data[data['Split'] == 'test']
data_dev = data[data['Split'] == 'dev']
data = data[data['Split'] == 'train']

# Ridit scoring annotations and confidence ratings
for attr in attributes:
    resp = attr_map[attr]
    resp_conf = attr_conf[attr]
    data[resp_conf + ".norm"] = data.groupby('Annotator.ID')[resp_conf].transform(ridit)
    data[resp + ".norm"] = data[resp].map(lambda x: 1 if x else -1) * data[resp_conf + ".norm"]
    data_dev[resp_conf + ".norm"] = data_dev.groupby('Annotator.ID')[resp_conf].transform(ridit)
    data_dev[resp + ".norm"] = data_dev[resp].map(lambda x: 1 if x else -1) * data_dev[resp_conf + ".norm"]

# Shuffle the data
data = shuffle(data)
data_dev = shuffle(data_dev)
data_test = shuffle(data_test)

# ELMO embeddings
options_file = args.embeddings + "options.json"
weight_file = args.embeddings + "weights.hdf5"
elmo = Elmo(options_file, weight_file, 1, dropout=0.5)

# pyTorch figures out device to do computation on
device_to_use = device("cuda:0" if is_available() else "cpu")

# Prepare all the inputs

x = [data['Structure'].values.tolist()[i:i + args.batch] for i in range(0, len(data['Structure']), args.batch)]
tokens = [data[token_col].values.tolist()[i:i + args.batch] for i in range(0, len(data[token_col]), args.batch)]
y = []
# loss_wts = {}

y = [{attr: (data[attr_map[attr] + ".norm"].values[i:i + args.batch]) for attr in attributes} for i in range(0, len(data[attr_map[attr] + ".norm"].values), args.batch)]
y[-1] = {attr: np.append(y[-1][attr], y[-2][attr][0:len(y[-2][attr]) - len(y[-1][attr])]) for attr in attributes}

# wt_raw = data[attr_conf[attr]].values
# loss_wts[attr] = [wt_raw[i:i + args.batch] for i in range(0, len(wt_raw), args.batch)]
# loss_wts[attr][-1] = np.append(loss_wts[attr][-1], loss_wts[attr][-2][0:len(loss_wts[attr][-2]) - len(loss_wts[attr][-1])])

x[-1] = x[-1] + x[-2][0:len(x[-2]) - len(x[-1])]    # last batch size hack
tokens[-1] = np.append(tokens[-1], tokens[-2][0:len(tokens[-2]) - len(tokens[-1])])

# Create dev data
data_dev_mean = data_dev.groupby('Unique.ID', as_index=False)[token_col, 'Is.Particular.norm', 'Is.Kind.norm', 'Is.Abstract.norm'].mean()
data_dev_mean['Structure'] = data_dev_mean['Unique.ID'].map(lambda x: data_dev[data_dev['Unique.ID'] == x]['Structure'].iloc[0])

dev_x = [data_dev_mean['Structure'].values.tolist()[i:i + args.batch] for i in range(0, len(data_dev_mean['Structure']), args.batch)]
dev_tokens = [data_dev_mean[token_col].values.tolist()[i:i + args.batch] for i in range(0, len(data_dev_mean[token_col]), args.batch)]
# loss_wts = {}

dev_y = {}
for attr in attributes:
    dev_y[attr] = [data_dev_mean[attr_map[attr] + ".norm"].values[i:i + args.batch] for i in range(0, len(data_dev_mean[attr_map[attr] + ".norm"].values), args.batch)]
    dev_y[attr][-1] = np.append(y[-1][attr], y[-2][attr][0:len(y[-2][attr]) - len(y[-1][attr])])

for attr in attributes:
    dev_y[attr] = np.concatenate(dev_y[attr], axis=None)
# wt_raw = data_dev_mean[attr_conf[attr]].values
# loss_wts[attr] = [wt_raw[i:i + args.batch] for i in range(0, len(wt_raw), args.batch)]
# loss_wts[attr][-1] = np.append(loss_wts[attr][-1], loss_wts[attr][-2][0:len(loss_wts[attr][-2]) - len(loss_wts[attr][-1])])

dev_x[-1] = dev_x[-1] + dev_x[-2][0:len(dev_x[-2]) - len(dev_x[-1])]    # last batch size hack
dev_tokens[-1] = np.append(dev_tokens[-1], dev_tokens[-2][0:len(dev_tokens[-2]) - len(dev_tokens[-1])])

# Initialise the model
trainer = MLPTrainer(embeddings=elmo, device=device_to_use,
                     attributes=attributes, attention=attention_type,
                     regressiontype=args.regressiontype)

# Training phase
trainer.fit(X=x, Y=y, tokens=tokens, verbosity=args.verbosity, dev=[dev_x, dev_y, dev_tokens])

# Save the model
