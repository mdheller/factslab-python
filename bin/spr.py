import pandas as pd
import numpy as np
from factslab.utility import ridit, dev_mode_group
from os.path import expanduser
from scipy.stats import spearmanr, pearsonr
pd.set_option('mode.chained_assignment', None)

if __name__ == "__main__":
    home = expanduser('~')
    attr_map = {"part": "Is.Particular", "kind": "Is.Kind", "abs": "Is.Abstract"}
    attr_conf = {"part": "Part.Confidence", "kind": "Kind.Confidence",
             "abs": "Abs.Confidence"}
    attrs = ["part", "kind", "abs"]
    attributes = ['Is.Particular.Norm', 'Is.Kind.Norm', 'Is.Abstract.Norm']

    datafile = home + "/Desktop/protocols/data/noun_raw_data_norm_122218.tsv"
    arg = pd.read_csv(datafile, sep="\t")

    arg['Sentence.ID'] = arg['Sentence.ID'].str.replace('sent_', '', regex=False)
    arg['Unique.ID'] = arg.apply(lambda x: str(x['Sentence.ID']) + "_" + str(x['Span']).split(',')[0] + "_" + str(x['Span']).split(',')[-1], axis=1)
    arg = arg.dropna()
    arg = arg[arg['Split'].isin(['train', 'dev'])]
    arg = arg.groupby('Unique.ID', as_index=True).mean()

    datafile_ = home + "/Desktop/protocols/data/spr/protoroles_eng_ud1.2_11082016.tsv"
    spr = pd.read_csv(datafile_, sep="\t")
    # pred token is 0 indexed in SPR
    spr['Unique.ID'] = spr.apply(lambda x: str(x['Sentence.ID']) + "_" + str(x["Arg.Tokens.Begin"]) + "_" + str(x["Arg.Tokens.End"]), axis=1)
    spr = spr[~spr['Is.Pilot']]
    spr = spr.dropna()

    spr = spr[spr['Split'].isin(['train', 'dev'])]

    # properties = ['change_of_location', 'instigation', 'partitive', 'was_for_benefit', 'existed_after', 'was_used', 'change_of_possession', 'existed_during', 'sentient', 'volition', 'change_of_state_continuous', 'awareness', 'existed_before', 'change_of_state']
    properties = ['volition', 'awareness', 'sentient', 'change_of_location', 'instigation', 'change_of_state', 'was_used', 'change_of_possession', 'partitive', 'was_for_benefit', 'existed_before', 'existed_during', 'existed_after']
    # arg_ids = list(set(arg['Unique.ID'].tolist()))
    print("Arg\n")
    for prop in properties:
        prop_df = spr[spr['Property'] == prop]
        prop_df.loc[:, 'Response.ridit'] = prop_df.groupby('Annotator.ID')['Response'].transform(ridit)
        prop_df = prop_df.groupby('Unique.ID', as_index=False).mean().reset_index(drop=True)
        prop_df = prop_df.loc[:, ['Unique.ID', 'Response.ridit']].dropna()

        for attr in attributes:
            prop_df.loc[:, attr] = prop_df['Unique.ID'].apply(lambda x: arg.loc[x][attr] if x in arg.index else None)
        prop_df = prop_df.dropna()

        print(prop.replace('_', ' '), '&', np.round(pearsonr(prop_df[attributes[0]].values, prop_df['Response.ridit'])[0], 2), '&', np.round(pearsonr(prop_df[attributes[1]].values, prop_df['Response.ridit'])[0], 2), '&', np.round(pearsonr(prop_df[attributes[2]].values, prop_df['Response.ridit'])[0], 2), "\\\\")

    attributes = ['Is.Particular.Norm', 'Is.Hypothetical.Norm', 'Is.Dynamic.Norm']
    datafile = home + "/Desktop/protocols/data/pred_raw_data_norm_122218.tsv"
    pred = pd.read_csv(datafile, sep="\t")
    pred['Sentence.ID'] = pred['Sentence.ID'].str.replace('sent_', '', regex=False)

    lst_col = 'Context.Span'
    x = pred.assign(**{lst_col: pred[lst_col].str.split(';')})
    pred = pd.DataFrame({col: np.repeat(x[col].values, x[lst_col].str.len()) for col in x.columns.difference([lst_col])}).assign(**{lst_col: np.concatenate(x[lst_col].values)})[x.columns.tolist()]
    pred['Unique.ID'] = pred.apply(lambda x: str(x['Sentence.ID']) + "_" + str(x['Context.Span']).split(',')[0] + "_" + str(x['Context.Span']).split(',')[-1], axis=1)
    pred = pred.dropna()
    pred = pred[pred['Split'].isin(['train', 'dev'])]
    pred = pred.groupby('Unique.ID', as_index=True).mean()

    # pred_ids = list(set(pred['Unique.ID'].tolist()))
    print("\nPred\n")
    for prop in properties:
        prop_df = spr[spr['Property'] == prop]
        prop_df.loc[:, 'Response.ridit'] = prop_df.groupby('Annotator.ID')['Response'].transform(ridit)
        prop_df = prop_df.groupby('Unique.ID', as_index=False).mean().reset_index(drop=True)
        prop_df = prop_df.loc[:, ['Unique.ID', 'Response.ridit']].dropna()
        for attr in attributes:
            prop_df.loc[:, attr] = prop_df['Unique.ID'].apply(lambda x: pred.loc[x][attr] if x in pred.index else None)
        prop_df = prop_df.dropna()

        print(prop.replace('_', ' '), '&', np.round(pearsonr(prop_df[attributes[0]].values, prop_df['Response.ridit'])[0], 2), '&', np.round(pearsonr(prop_df[attributes[1]].values, prop_df['Response.ridit'])[0], 2), '&', np.round(pearsonr(prop_df[attributes[2]].values, prop_df['Response.ridit'])[0], 2), "\\\\")
