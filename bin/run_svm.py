import argparse
import numpy as np
import pandas as pd
# from factslab.utility import load_glove_embedding
from sklearn.svm import SVC
# from factslab.datafeatures import DependencyTree
from math import sqrt
# from allennlp.modules.elmo import Elmo, batch_to_ids
from predpatt import load_conllu
from predpatt import PredPatt
from predpatt import PredPattOpts
from os.path import expanduser


def noun2features(sent_feat, token):
    '''Extract features from a word'''
    sent = sent_feat[0]
    feats = sent_feat[1]
    words = sent.tokens[token].text
    deps = [x[2].tag for x in sent.tokens[token].dependents]
    deps_text = [x[2].text for x in sent.tokens[token].dependents]
    features = [int('Number=Plur' in feats[token]),        # plural
                int('DET' in deps),                        # has determiner
                int('the' in deps_text),              # has 'the' determiner
                int('some' in deps_text),             # has 'some' quantifier
                int('many' in deps_text),             # has 'many' quantifier
                int('few' in deps_text),              # has 'few' quantifier
                int('Definite=Ind' in feats[token]),  # Indefinite
                int('Definite=Def' in feats[token])   # Definite
                ]
    return np.array(features)


def pred2features(sent_feat, token):
    '''Extract features from a word'''
    sent = sent_feat[0]
    feats = sent_feat[1]
    words = sent.tokens[token].text
    deps = [x[2].tag for x in sent.tokens[token].dependents]
    deps_text = [x[2].text for x in sent.tokens[token].dependents]
    modals = ['can', 'could', 'should', 'would']
    features = [int('Tense=Past' in feats[token]),       # Past tense
                int('Tense=Pres' in feats[token]),       # Present tense
                int('VerbForm=Fin' in feats[token]),     # Finitive
                int('VerbForm=Inf' in feats[token]),     # Infinitive
                int('VerbForm=Part' in feats[token]),    # Participle
                int('VerbForm=Ger' in feats[token]),     # Gerund
                int('AUX' in deps),                      # Auxillary
                int(bool(set(modals) & set(deps_text)))  # Modals
                ]
    return np.array(features)


# initialize argument parser
description = 'Run an RNN regression on Genericity protocol annotation.'
parser = argparse.ArgumentParser(description=description)

parser.add_argument('--prot',
                    type=str,
                    default="noun")
parser.add_argument('--features',
                    type=str,
                    default='features.tsv')
parser.add_argument('--embeddings',
                    type=str,
                    default='../../../../Downloads/embeddings/glove.42B.300d')
parser.add_argument('--regressiontype',
                    type=str,
                    default="linear")

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
    features_func = noun2features
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
    features_func = pred2features

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

# Load the features
features = {}
# vocab = []

# Don't read_csv the features file. read_csv can't handle quotes
with open(args.features, 'r') as f:
    for line in f.readlines():
        feats = line.split('\t')
        # features[feats[0]] = DependencyTree.fromstring(feats[1])
        features[feats[0]] = feats[1].split()
        # vocab.append(feats[2].split())

# Load the predpatt objects for creating features

files = ['/UD_English-r1.2/en-ud-train.conllu',
         '/UD_English-r1.2/en-ud-dev.conllu',
         '/UD_English-r1.2/en-ud-test.conllu']
home = expanduser("~/Downloads/")
options = PredPattOpts(resolve_relcl=True, borrow_arg_for_relcl=True, resolve_conj=False, cut=True)  # Resolve relative clause
patt = {}

for file in files:
    path = home + file
    with open(path, 'r') as infile:
        for sent_id, ud_parse in load_conllu(infile.read()):
            patt[file[17:] + " " + sent_id] = PredPatt(ud_parse, opts=options)

data['Structure'] = data['Sentence.ID'].map(lambda x: (patt[x], features[x]))

# vocab = list(set(sum(vocab, [])))
data_dev = data[data['Split'] == 'dev']
data = data[data['Split'] == 'train']

# load the glove embedding
# embeddings = load_glove_embedding(args.embeddings, vocab)
# ELMO embeddings
# options_file = "/Users/venkat/Downloads/embeddings/options.json"
# weight_file = "/Users/venkat/Downloads/embeddings/weights.hdf5"
# elmo = Elmo(options_file, weight_file, 1, dropout=0.5)

# Shuffle the data
data = data.sample(frac=1).reset_index(drop=True)
data_dev = data_dev.sample(frac=1).reset_index(drop=True)
data_test = data_test.sample(frac=1).reset_index(drop=True)
# Define attributes for regression

raw_x = [struct for struct in data['Structure']]
tokens = data[token_col].tolist()

x = []
x = np.array([features_func(sent, token) for sent, token in zip(raw_x, tokens)])
y = {}
loss_wts = {}

for attr in attributes:
    y[attr] = np.array(data[attr_map[attr]].tolist())
    loss_wts[attr] = np.array(data[attr_conf[attr]].tolist())
    y[attr] = y[attr]

# Initialise dev data
data_dev = pd.read_csv(dev_datafile, sep="\t")
data_dev['Structure'] = data_dev['Sentence.ID'].map(lambda x: (patt[x], features[x]))

raw_dev_x = [struct for struct in data_dev['Structure']]
dev_tokens = data_dev[token_col].tolist()

dev_x = np.array([features_func(sent, token) for sent, token in zip(raw_dev_x, dev_tokens)])
dev_y = {}
dev_wts = {}

for attr in attributes:
    dev_y[attr] = data_dev[attr_map[attr]].tolist()
    dev_wts[attr] = data_dev[attr_conf[attr]].tolist()

for attr in attributes:
    # train the model
    trainer = SVC()
    trainer.fit(x, y[attr], sample_weight=loss_wts[attr])

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
