import numpy as np
import pandas as pd


datafile = "noun_long_data.tsv"
dev_datafile = "noun_data_dev.tsv"
response = ["Is.Particular", "Is.Kind", "Is.Abstract"]
response_conf = ["Part.Confidence", "Kind.Confidence", "Abs.Confidence"]
attributes = ["part", "kind", "abs"]
attr_map = {"part": "Is.Particular", "kind": "Is.Kind", "abs": "Is.Abstract"}
attr_conf = {"part": "Part.Confidence", "kind": "Kind.Confidence",
         "abs": "Abs.Confidence"}
token_col = "Noun.Token"

data = pd.read_csv(datafile, sep="\t")

data['SentenceID.Token'] = data['Sentence.ID'].map(lambda x: x) + "_" + data[token_col].map(lambda x: str(x))

# Split the datasets into train, dev, test
data_test = data[data['Split'] == 'test']
data = data[data['Split'] != 'test']
data = data[data['Split'] != 'dev']

# Convert responses to 1s and 0s
for resp in response:
    data[resp] = data[resp].astype(int)

# convert response confs to logit ridit scores
for resp in response_conf:
    data[resp] = data.groupby('Annotator.ID')[resp].apply(lambda x: x.rank() / (len(x) + 1.))
    data[resp] = np.log(data[resp]) - np.log(1. - data[resp])

concreteness = pd.read_csv('concreteness.tsv', sep="\t")

concreteness = concreteness.loc[:, ['Word', 'Conc.M']]

import ipdb; ipdb.set_trace()
data['conc'] = data['Noun'].map(lambda x: concreteness[concreteness['Word'] == x]['Conc.M'] if concreteness['Word'].str.contains(x).any() else -1)
