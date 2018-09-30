import argparse
import numpy as np
import pandas as pd
# from factslab.utility import load_glove_embedding
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from math import sqrt
# from allennlp.modules.elmo import Elmo, batch_to_ids
from predpatt import load_conllu
from predpatt import PredPatt
from predpatt import PredPattOpts
from os.path import expanduser
from scipy.stats import spearmanr
from statistics import median
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


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


def noun2features(sent_feat, token):
    '''Extract features from a word'''
    sent = sent_feat[0]
    feats = sent_feat[1]
    # words = sent.tokens[token].text
    deps = [x[2].tag for x in sent.tokens[token].dependents]
    deps_text = [x[2].text for x in sent.tokens[token].dependents]
    features = [int('Number=Plur' in feats[token]),   # plural 0
                int('Number=Sing' in feats[token]),   # singular 1
                int('DET' in deps),                   # has determiner 2
                int('the' in deps_text),              # has 'the' determiner 3
                int('some' in deps_text),             # has 'some' quantifier 4
                int('many' in deps_text),             # has 'many' quantifier 5
                int('few' in deps_text),              # has 'few' quantifier 6
                int('Definite=Ind' in feats[token]),  # Indefinite 7
                int('Definite=Def' in feats[token]),  # Definite 8
                int('Gender=Masc' in feats[token]),   # Masculine 9
                int('Gender=Fem' in feats[token]),    # Feminine 10
                int('Gender=Neut' in feats[token]),   # Neutral 11
                int('PronType=Prs' in feats[token]),  # personal pronouns 12
                int('PronType=Art' in feats[token]),  # article 13
                int('Poss=Yes' in feats[token]),      # Possessive 14
                int('Abbr=Yes' in feats[token])       # Abbreviation 15
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

if __name__=="__main__":
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
        datafile = "../../../protocols/data/noun_long_data.tsv"
        # dev_datafile = "noun_data_dev.tsv"
        response = ["Is.Particular", "Is.Kind", "Is.Abstract"]
        response_conf = ["Part.Confidence", "Kind.Confidence", "Abs.Confidence"]
        attributes = ["part", "kind", "abs"]
        attr_map = {"part": "Is.Particular", "kind": "Is.Kind", "abs": "Is.Abstract"}
        attr_conf = {"part": "Part.Confidence", "kind": "Kind.Confidence",
                 "abs": "Abs.Confidence"}
        token_col = "Noun.Token"
        features_func = noun2features
    else:
        datafile = "../../../protocols/data/pred_long_data.tsv"
        # dev_datafile = "pred_data_dev.tsv"
        response = ["Is.Particular", "Is.Hypothetical", "Is.Dynamic"]
        response_conf = ["Part.Confidence", "Hyp.Confidence", "Dyn.Confidence"]
        attributes = ["part", "hyp", "dyn"]
        attr_map = {"part": "Is.Particular", "dyn": "Is.Dynamic", "hyp": "Is.Hypothetical"}
        attr_conf = {"part": "Part.Confidence", "dyn": "Dyn.Confidence",
                 "hyp": "Hyp.Confidence"}
        token_col = "Pred.Root.Token"
        features_func = noun2features

    data = pd.read_csv(datafile, sep="\t")

    data['SentenceID.Token'] = data['Split'].map(lambda x: x) + data['Sentence.ID'].map(lambda x: str(x)) + "_" + data[token_col].map(lambda x: str(x))

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
                patt[file[23:][:-7] + " " + sent_id] = PredPatt(ud_parse, opts=options)

    data['Structure'] = data['Sentence.ID'].map(lambda x: (patt['train sent_' + str(x)], features['train sent_' + str(x)]))

    # vocab = list(set(sum(vocab, [])))
    # Split the datasets into train, dev, test
    data_test = data[data['Split'] == 'test']
    data_dev = data[data['Split'] == 'dev']
    data = data[data['Split'] == 'train']

    # Ridit scoring annotations and confidence ratings
    for attr in attributes:
        resp = attr_map[attr]
        resp_conf = attr_conf[attr]
        data[resp_conf + ".norm"] = data.groupby('Annotator.ID')[resp_conf].transform(ridit)
        data[resp + ".norm"] = data[resp].map(lambda x: 1 if x else 0)
        data_dev[resp_conf + ".norm"] = data_dev.groupby('Annotator.ID')[resp_conf].transform(ridit)
        data_dev[resp + ".norm"] = data_dev[resp].map(lambda x: 1 if x else 0)

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

    for attr in attributes:
        y[attr] = np.array(data[attr_map[attr] + ".norm"].tolist())

    # data_dev_mean = data_dev.groupby('SentenceID.Token', as_index=False)['Is.Particular.norm', 'Is.Kind.norm', 'Is.Abstract.norm'].mean()
    # raw_dev_x = [data_dev[data_dev['SentenceID.Token'] == sent_id]['Structure'].values[0] for sent_id in data_dev_mean['SentenceID.Token']]
    # dev_tokens = [data_dev[data_dev['SentenceID.Token'] == sent_id][token_col].values[0] for sent_id in data_dev_mean['SentenceID.Token']]

    # dev_x = np.array([features_func(sent, token) for sent, token in zip(raw_dev_x, dev_tokens)])
    dev_y = {}
    parameters = {}
    for attr in attributes:
        # dev_y[attr] = data_dev_mean[attr_map[attr] + ".norm"].tolist()
        parameters[attr] = {'C': [50, 100, 200], 'epsilon': [0.01, 0.35],
                  'sample_weight': np.array(data[attr_conf[attr] + ".norm"].values.tolist())}
    import ipdb; ipdb.set_trace()
    for attr in attributes:
        print(attr_map[attr])
        # median = data[attr_map[attr] + ".norm"].median()
        mean = y[attr].mean()
        # train the model
        trainer = SVC()
        # trainer.fit(x, y[attr])
        clf = GridSearchCV(trainer, parameters[attr], return_train_score=True,
                           n_jobs=10, verbose=1)
        clf.fit(x, y[attr])
        # y_ = trainer.predict(x)
        # ir = StandardScaler()
        # # y_ = ir.fit_transform(y_, dev_y[attr])
        # print("MAE with median:", mae(dev_y[attr], [data[attr_map[attr] + ".norm"].median() for a in range(len(dev_y[attr]))]))
        # print("MAE:", mae(y[attr], y_))
        # print("Spearman:", spearmanr(y[attr], y_)[0], "\n")
        # print("R-sqared:", 1 - (mse(y[attr], y_) / mse(y[attr], [mean for a in range(len(y[attr]))])))
        for p, tr, trr in zip(clf.cv_results_['params'], clf.cv_results_['mean_test_score'], clf.cv_results_['mean_train_score']):
            print(p, tr, trr)

        # conf_mat = np.zeros((2, 2))
        # for i, output in enumerate(predictions):
        #     target = dev_y[attr][i]
        #     conf_mat[int(output)][int(target)] += 1
        # print(attr)
        # print(conf_mat)
        # # Accuracy
        # accuracy = sum([conf_mat[i][i] for i in range(2)]) / np.sum(conf_mat)

        # # Precision
        # p_macro = np.array([conf_mat[i][i] for i in range(2)]) / np.array([sum([conf_mat[j][i] for j in range(2)]) for i in range(2)])
        # # Recall
        # r_macro = np.array([conf_mat[i][i] for i in range(2)]) / np.array([sum([conf_mat[i][j] for j in range(2)]) for i in range(2)])

        # TP = conf_mat[1][1]
        # TN = conf_mat[0][0]
        # FP = conf_mat[1][0]
        # FN = conf_mat[0][1]
        # matthews_corr = ((TP * TN) - (FP * FN)) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        # # F1 Score
        # f1 = np.sum(2 * (p_macro * r_macro) / (p_macro + r_macro)) / 2
        # macro_acc = np.sum(p_macro) / 2
        # print("Micro Accuracy:", accuracy)
        # print("Macro Accuracy:", macro_acc)
        # print("Macro F1 score:", f1)
        # print("Matthews correlation coefficient:", matthews_corr)
