import re
import numpy as np
import pandas as pd
from collections import defaultdict
from pyparsing import nestedExpr


class LexicalConceptualStructureLexicon(object):

    def __init__(self, filename):
        entries = defaultdict(list)
        curr_entry = {}

        for l in open(filename):

            l = l.strip()

            # if the *only* thing in the line is "(" don't want to
            # miss multi-line values by saying l[0] == '('
            if not l or l == '(':
                continue

            # we have stripped off the leading space so we just need
            # to check for the first character
            elif l[0] == ':':
                curr_attr, curr_val = re.findall(':(.+?)\s(.+)', l)[0]

                curr_attr = curr_attr.replace('_', '').lower()
                curr_attr = 'wordclass' if curr_attr == 'class' else curr_attr
                curr_attr = 'word' if curr_attr == 'defword' else curr_attr

                curr_entry[curr_attr] = curr_val.replace('"', '')

            # same as for entry opener
            elif l == ')':
                entry = LexicalConceptualStructureLexiconEntry(**curr_entry)
                entries[entry.word].append(entry)

            # if the line isn't a comment, the only remaining thing it
            # could be is the non-first line in a multi-line value
            elif l[0] != ';':
                curr_val = l
                curr_entry[curr_attr] += curr_val.replace('"', '')

        self._entries = dict(entries)

    def __getitem__(self, word):
        return self._entries[word]

    @property
    def verbs(self):
        return list(self._entries.keys())

    def stative(self, word):
        return [entry.stative for entry in self._entries[word]]

    def eventive(self, word):
        return [entry.eventive for entry in self._entries[word]]


class LexicalConceptualStructureLexiconEntry(object):

    def __init__(self, word, wordclass, wnsense, propbank, thetaroles,
                 lcs, gloss=None, glosshead=None, varspec=None, roman=None):

        parser = nestedExpr('(',')')
        parse = lambda string: parser.parseString(string).asList()

        self.word = word
        self.gloss = gloss
        self.glosshead = glosshead
        self.roman = roman
        self.wordclass = wordclass
        self.wnsense = parse(wnsense)
        self.propbank = parse(propbank)
        self.thetaroles = parse(thetaroles)
        self.lcs = parse(lcs)[0]
        self.varspec = parse(varspec) if varspec is not None else varspec

    @property
    def stative(self):
        return self.lcs[0] == 'be'

    @property
    def eventive(self):
        return self.lcs[0] != 'be'


def eventivity(row):
    global lcs
    x = row['lcs_eventive']
    dyn = row['Is.Dynamic']
    if x.count(x[0]) == len(x):
        return int(dyn == int(x[0]))
    else:
        return dyn


lcs = LexicalConceptualStructureLexicon('verbs-English.lcs')

# Read annotations
datafile = "pred_long_data.tsv"
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
data = data[data['Split'] != 'dev']

# Convert responses to 1s and 0s
for resp in response:
    data[resp] = data[resp].astype(int)

# convert response confs to logit ridit scores
for resp in response_conf:
    data[resp] = data.groupby('Annotator.ID')[resp].apply(lambda x: x.rank() / (len(x) + 1.))
    data[resp] = np.log(data[resp]) - np.log(1. - data[resp])


dyn_check = data.loc[:, ['Predicate.Lemma', 'Is.Dynamic']]

dyn_check['lcs_eventive'] = dyn_check['Predicate.Lemma'].map(lambda x: lcs.eventive(x) if x in lcs.verbs else -1)

dyn_check = dyn_check[dyn_check['lcs_eventive'] != -1]

dyn_check['compare'] = dyn_check['lcs_eventive'].map(lambda x: 1 if x.count(x[0]) != len(x) else int(x[0]))

dyn_check['compare'] = dyn_check.apply(lambda row: eventivity(row), axis=1)
print("LCS", sum(dyn_check['compare']) / len(dyn_check))

concreteness = pd.read_csv('concreteness.tsv', sep="\t")
