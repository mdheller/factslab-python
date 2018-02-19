import numpy as np
import pandas as pd

from factslab.utility import load_glove_embedding
from factslab.datastructures import ConstituencyTree
from factslab.pytorch.childsumtreelstm import ChildSumConstituencyTreeLSTM
from factslab.pytorch.rnnregression import RNNRegressionTrainer

data = pd.read_csv('../../factslab-data/megaattitude/megaattitude_v1.csv')

# remove subjects that are marked for exclusion
data = data[~data.exclude]

# remove null responses; removes 10 lines
data = data[~data.response.isnull()]

# the intransitive frame is denoted by an empty string, so make it overt
data.loc[data.frame.isnull(),'frame'] = 'null'

# convert responses to logit ridit scores
data['response'] = data.groupby('participant').response.apply(lambda x: x.rank()/(len(x)+1.))
data['response'] = np.log(data.response) - np.log(1.-data.response)

# convert "email" to "e-mail" to deal with differences between
# megaattitude_v1.csv and structures.tsv
data['condition'] = data.verb.replace('email', 'e-mail')+'-'+data.frame+'-'+data.voice

# load structures into a dictionary
with open('../../factslab-data/megaattitude/structures.tsv') as f:
    structures = dict([line.replace(',', 'COMMA').strip().split('\t') for line in f])

    structures = {k: ConstituencyTree.fromstring(s) for k, s in structures.items()}

# get the structure IDs from the dictionary keys
conditions = list(structures.keys())

# filter down to those conditions found in conditions
data = data[data.condition.isin(conditions)]

# build the vocab list up from the structures
vocab = list({word
              for tree in structures.values()
              for word in tree.leaves()})

# load the glove embedding
embeddings = load_glove_embedding('../../../embeddings/glove/glove.42B.300d', vocab)

# train the model
trainer = RNNRegressionTrainer(embeddings=embeddings,
                               rnn_classes=ChildSumConstituencyTreeLSTM,
                               rnn_hidden_sizes=300)
trainer.fit(structures=[[structures[c] for c in data.condition.values]],
            targets=data.response.values,
            lr=1., batch_size=1000)

# trainer = RNNRegressionTrainer(embeddings=embeddings, gpu=True,
#                                rnn_classes=[LSTM, ChildSumConstituencyTreeLSTM],
#                                rnn_hidden_sizes=300)    
# trainer.fit(structures=[[structures[c].words() for c in data.condition.values],
#                         [structures[c] for c in data.condition.values]],
#             targets=data.response.values,
#             lr=1.)
