'''
    Extracts all features(lexical, morphological, syntactic, semantic) from UD
    and prints to a TSV file in a neat format.
    Sentence.ID    [Word1feats Word2feats Word3Feats]   [Lemma1 Lemma2 Lemma3]
'''
from os.path import expanduser


files = ['/Downloads/UD_English-r1.2/en-ud-train.conllu',
         '/Downloads/UD_English-r1.2/en-ud-dev.conllu',
         '/Downloads/UD_English-r1.2/en-ud-test.conllu']
home = expanduser("~")

for file in files:
    with open(home + '/Research/protocols/data/features-2.tsv', 'a') as fout:
        path = home + file
        with open(path, 'r') as f:
            id = 0
            feats = []
            lemmas = []
            for line in f:
                if line != "\n":
                    all_feats = line.split("\t")
                    feats.append("UPOS=" + all_feats[3] + "|" + "XPOS=" + all_feats[4] + "|" + all_feats[5] + "|" + "DEPREL=" + all_feats[7])
                    lemmas.append(all_feats[2])
                else:
                    id += 1
                    sent_id = file[33:][:-7] + " sent_" + str(id)
                    feats = " ".join(feats)
                    lemmas = " ".join(lemmas)
                    fout.write(sent_id + "\t" + feats + "\t" + lemmas + "\n")
                    feats = []
                    lemmas = []
