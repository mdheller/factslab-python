# A script to print sentence ids and corresponding sentence
# from nltk import DependencyGraph
import re
from os.path import expanduser
from os import remove


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

# Delete file if it already exists since it'll be opened in append mode
struct_path = expanduser("~/Desktop") + "/protocols/data/structures.tsv"
remove(struct_path)

for file in files:
    path = home + file
    with open(struct_path, 'a') as fout:
        with open(path, 'r') as f:
            idNo = 0
            a = ""
            words = []
            for line in f:
                if line != "\n":
                    a += line
                    words.append(line.split("\t")[1])
                else:
                    idNo += 1
                    a = html_ify(a)
                    # structure = DependencyGraph(a, top_relation_label='root')
                    sent = " ".join(words)
                    sent = html_ify(sent)
                    sent_id = file[23:][:-7] + " sent_" + str(idNo)
                    # structures.append(structure)
                    a = ""
                    words = []
                    fout.write(sent_id + "\t" + sent + "\n")
