from os.path import expanduser


files = ['/UD_English-r1.2/en-ud-train.conllu',
         '/UD_English-r1.2/en-ud-dev.conllu',
         '/UD_English-r1.2/en-ud-test.conllu']
home = expanduser("~/Downloads/")

for file in files:
    with open('features.tsv', 'a') as fout:
        path = home + file
        with open(path, 'r') as f:
            id = 0
            feats = []
            for line in f:
                if line != "\n":
                    all_feats = line.split("\t")
                    feats.append("UPOS=" + all_feats[3] + "|" + "XPOS=" + all_feats[4] + "|" + all_feats[5] + "|" + "DEPREL=" + all_feats[7])
                else:
                    id += 1
                    sent_id = file[23:][:-7] + " sent_" + str(id)
                    feats = " ".join(feats)
                    fout.write(sent_id + "\t" + feats + "\n")
                    feats = []
