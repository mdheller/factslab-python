# A script to print convert all UD conllu
from nltk import DependencyGraph
import re


def html_ify(s):
    '''
        Takes care of &quot &lsqb &rsqb &#39
    '''
    html_string = re.sub(r'\)', r'&rcrb;', s)
    html_string = re.sub(r'\(', r'&lcrb;', html_string)
    return html_string


files = ['en-ud-train.conllu', 'en-ud-dev.conllu', 'en-ud-test.conllu']
structures = []
for file in files:
    with open('structures.tsv', 'a') as fout:
        with open(file, 'r') as f:
            id = 0
            a = ""
            words = []
            for line in f:
                if line != "\n":
                    a += line
                    words.append(line.split("\t")[1])
                else:
                    id += 1
                    a = html_ify(a)
                    structure = DependencyGraph(a, top_relation_label='root')
                    sent = " ".join(words)
                    sent = html_ify(sent)
                    sent_id = file + " sent_" + str(id)
                    structures.append(structure)
                    a = ""
                    words = []
                    fout.write(sent_id + "\t" + " ".join(str(structures[-1].tree()).splitlines()) + "\t" + sent + "\n")