import pandas as pd
import numpy as np
from os.path import expanduser
from scipy.stats import spearmanr, pearsonr

if __name__ == "__main__":
    home = expanduser('~')

    datafile = home + "/Desktop/protocols/data/pred_raw_data_norm_122218.tsv"
    pred_data = pd.read_csv(datafile, sep="\t")

    datafile_ = home + "/Desktop/protocols/data/it-happened_eng_ud1.2_07092017_normalized.tsv"
    fact_data = pd.read_csv(datafile_, sep="\t")

    pred_data = pred_data[pred_data['Split'].isin(['train', 'dev'])]
    pred_data['Sentence.ID'] = pred_data['Sentence.ID'].str.replace('sent_', '', regex=False)
    pred_data['Unique.ID'] = pred_data.apply(lambda x: str(x['Sentence.ID']) + "_" + str(x["Root.Token"]), axis=1)
    pred_data = pred_data.groupby('Unique.ID', as_index=False).mean().reset_index(drop=True)

    fact_data = fact_data[fact_data['Split'].isin(['train', 'dev'])]
    fact_data['Unique.ID'] = fact_data.apply(lambda x: str(x['Sentence.ID']) + "_" + str(x["Pred.Token"] - 1), axis=1)
    fact_data = fact_data.groupby('Unique.ID', as_index=False).mean().reset_index(drop=True)

    hyp_fact = pred_data.loc[:, ['Unique.ID', 'Is.Hypothetical.Norm']]
    fact_ids = fact_data['Unique.ID'].tolist()
    hyp_fact['Happened.Norm'] = hyp_fact['Unique.ID'].apply(lambda x: fact_data[fact_data['Unique.ID'] == x]['Happened.Norm'].iloc[0] if x in fact_ids else None)
    hyp_fact2 = hyp_fact.dropna()
    print(np.round(len(hyp_fact2) / len(hyp_fact), 2))
    print("Spearman correlation: ", np.round(spearmanr(hyp_fact2['Is.Hypothetical.Norm'].values, hyp_fact2['Happened.Norm'].values)[0], 2))
    print("Pearson correlation: ", np.round(pearsonr(hyp_fact2['Is.Hypothetical.Norm'].values, hyp_fact2['Happened.Norm'].values)[0], 2))
