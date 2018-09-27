# A script to print sentence ids and corresponding sentence
# from nltk import DependencyGraph
import re
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
    path = home + file
    with open('../../../../protocols/data/structures.tsv', 'a') as fout:
        with open(path, 'r') as f:
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
                    # structure = DependencyGraph(a, top_relation_label='root')
                    sent = " ".join(words)
                    sent = html_ify(sent)
                    sent_id = file + " sent_" + str(id)
                    # structures.append(structure)
                    a = ""
                    words = []
                    fout.write(sent_id + "\t" + sent + "\n")
