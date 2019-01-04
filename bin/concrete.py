import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from os.path import expanduser


if __name__ == "__main__":
    home = expanduser('~')
    datafile = home + '/Desktop/protocols/data/noun_raw_data_norm_122218.tsv'

    data = pd.read_csv(datafile, sep="\t")

    # Split the datasets into train, dev, test
    data = data[data['Split'].isin(['train', 'dev'])]

    path = home + "/Desktop/protocols/data/concreteness.tsv"
    concreteness = pd.read_csv(path, sep="\t")
    list_of_lemmas = concreteness['Word'].values.tolist()

    abs_conc = data.groupby('Lemma')['Is.Abstract.Norm'].mean().to_frame().reset_index()
    abs_conc['Concreteness'] = abs_conc['Lemma'].map(lambda x: concreteness[concreteness['Word'] == x.lower()]['Conc.M'].values[0] if x.lower() in list_of_lemmas else -1)
    ini = len(abs_conc)
    abs_conc = abs_conc[abs_conc['Concreteness'] != -1]
    print(len(abs_conc) / ini)
    print("Spearman correlation: ", np.round(spearmanr(abs_conc['Is.Abstract.Norm'].values, abs_conc['Concreteness'].values)[0], 2))
    print("Pearson correlation: ", np.round(pearsonr(abs_conc['Is.Abstract.Norm'].values, abs_conc['Concreteness'].values)[0], 2))
