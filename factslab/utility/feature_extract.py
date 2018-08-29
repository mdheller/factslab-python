from nltk import DependencyGraph
import re
from predpatt import load_conllu
from predpatt import PredPatt
from predpatt import PredPattOpts
import pickle
from os.path import expanduser


def html_ify(s):
    '''
        Takes care of &quot &lsqb &rsqb &#39
    '''
    html_string = re.sub(r'\)', r'&rcrb;', s)
    html_string = re.sub(r'\(', r'&lcrb;', html_string)
    return html_string


files = ['/UD_English-r1.2/en-ud-train.conllu',
         '/UD_English-r1.2/en-ud-dev.conllu',
         '/UD_English-r1.2/en-ud-test.conllu']
home = expanduser("~/Downloads/")


structures = []
for file in files:
    with open('features.tsv', 'a') as fout:
        path = home + file
        with open(path, 'r') as f:
            id = 0
            a = ""
            words = []
            feats = []
            for line in f:
                if line != "\n":
                    a += line
                    words.append(line.split("\t")[1])
                    feats.append(line.split("\t")[5])
                else:
                    id += 1
                    sent_id = file[17:] + " sent_" + str(id)
                    feats = " ".join(feats)
                    fout.write(sent_id + "\t" + feats + "\n")
                    words = []
                    feats = []
