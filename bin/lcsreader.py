import re
import pandas as pd
from collections import defaultdict
from pyparsing import nestedExpr
from os.path import expanduser
import pickle

class LexicalConceptualStructureLexicon(object):

    def __init__(self, filename):
        entries = defaultdict(list)
        curr_entry = {}

        for line in open(filename):

            line = line.strip()

            # if the *only* thing in the line is "(" don't want to
            # miss multi-line values by saying l[0] == '('
            if not line or line == '(':
                continue

            # we have stripped off the leading space so we just need
            # to check for the first character
            elif line[0] == ':':
                curr_attr, curr_val = re.findall(':(.+?)\s(.+)', line)[0]

                curr_attr = curr_attr.replace('_', '').lower()
                curr_attr = 'wordclass' if curr_attr == 'class' else curr_attr
                curr_attr = 'word' if curr_attr == 'defword' else curr_attr

                curr_entry[curr_attr] = curr_val.replace('"', '')

            # same as for entry opener
            elif line == ')':
                entry = LexicalConceptualStructureLexiconEntry(**curr_entry)
                entries[entry.word].append(entry)

            # if the line isn't a comment, the only remaining thing it
            # could be is the non-first line in a multi-line value
            elif line[0] != ';':
                curr_val = line
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
    x = [int(a) for a in row['lcs_eventive']]
    dyn = row['Is.Dynamic.Norm']
    if dyn in x:
        return 1
    else:
        return 0


if __name__ == "__main__":
    home = expanduser('~')
    path = home + "/Desktop/protocols/data/verbs-English.lcs"
    # lcs = LexicalConceptualStructureLexicon(path)
    # with open(home + '/Downloads/lcs.pkl', 'wb') as f:
    #     pickle.dump(lcs, f)
    # print("Pickled")
    with open(home + '/Downloads/lcs.pkl', 'rb') as f:
        lcs = pickle.load(f)
    # Read annotations
    datafile = home + "/Desktop/protocols/data/pred_raw_data_norm_122218.tsv"

    data = pd.read_csv(datafile, sep="\t")

    # Split the datasets into train, dev, test
    data = data[data['Split'].isin(['train', 'dev'])]

    dyn_lcs = data.groupby('Lemma')['Is.Dynamic.Norm'].apply(list).to_frame().reset_index()
    dyn_lcs['lcs'] = dyn_lcs['Lemma'].map(lambda x: lcs.eventive(x.lower()) if x.lower() in lcs.verbs else -1)
    num_of_lemmas = len(dyn_lcs)
    dyn_lcs = dyn_lcs[dyn_lcs['lcs'] != -1]
    dyn_lcs.set_index('Lemma', inplace=True)
    dyn_lcs['dyn'] = dyn_lcs['Is.Dynamic.Norm'].apply(lambda x: [a > 0 for a in x])
    dyn_lcs['comp'] = dyn_lcs.apply(lambda x: 1 if set(x['dyn']).intersection(set(x['lcs'])) else 0, axis=1)

    print("Percentage of lemmas found in lcs database:", len(dyn_lcs) / num_of_lemmas)
    print("They share at least one sense:", sum(dyn_lcs['comp']) / len(dyn_lcs))
