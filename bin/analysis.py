import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from factslab.utility import r1_score
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns


sigdig = 1
data_arg = pd.read_csv('dev_preds_arg.tsv', sep='\t')
data_pred = pd.read_csv('dev_preds_pred.tsv', sep='\t')

# Do analysis on argument over here
# print(data_arg[data_arg['kind.Pred']>1]['kind.Pred'])
thing_words = data_arg[data_arg['Lemma'].str.contains('thing')]
# things_words_pred = data_pred[data_pred['Lemma'].str.contains('thing')]
# print(set(thing_words["Lemma"].tolist()))

plt.figure()
sns.distplot(thing_words['Is.Particular.Norm'], hist=False, label='Part Ann').get_figure()
sns.distplot(thing_words['part.Pred'], hist=False, label='Part Pred').get_figure()
sns.distplot(thing_words['Is.Kind.Norm'], hist=False, label='Kind Ann').get_figure()
sns.distplot(thing_words['kind.Pred'], hist=False, label='Kind Pred').get_figure()
sns.distplot(thing_words['Is.Abstract.Norm'], hist=False, label='Abs Ann').get_figure()
sns.distplot(thing_words['abs.Pred'], hist=False, label='Abs Pred').get_figure()
plt.xlabel('Normalized score')
plt.savefig('things.png', transparent=True)


# print([(a, len(data_arg[data_arg['POS'] == a])) for a in list(set(data_arg['POS'].tolist()))])
# # R1 and correlation based on POS and gov_rel
# print("\nArg POS")
# for pos in list(set(data_arg['POS'].tolist())):
#     data_new = data_arg[data_arg['POS'] == pos]
#     print(pos, '&', np.round(pearsonr(data_new['Is.Particular.Norm'], data_new['part.Pred'])[0] * 100, sigdig), '&', np.round(r1_score(data_new['Is.Particular.Norm'], data_new['part.Pred']) * 100, sigdig), '&', np.round(pearsonr(data_new['Is.Kind.Norm'], data_new['kind.Pred'])[0] * 100, sigdig), '&', np.round(r1_score(data_new['Is.Kind.Norm'], data_new['kind.Pred']) * 100, sigdig), '&', np.round(pearsonr(data_new['Is.Abstract.Norm'], data_new['abs.Pred'])[0] * 100, sigdig), '&', np.round(r1_score(data_new['Is.Abstract.Norm'], data_new['abs.Pred']) * 100, sigdig), '&', np.round(r1_score(data_new.loc[:, ['Is.Particular.Norm', 'Is.Kind.Norm', 'Is.Abstract.Norm']].values, data_new.loc[:, ['part.Pred', 'kind.Pred', 'abs.Pred']].values) * 100, sigdig), "\\\\")

# print([(a, len(data_arg[data_arg['DEPREL'] == a])) for a in list(set(data_arg['DEPREL'].tolist()))])
# print("\nArg DEPREL")
# # R1 and correlation based on POS and gov_rel
# for deprel in list(set(data_arg['DEPREL'].tolist())):
#     data_new = data_arg[data_arg['DEPREL'] == deprel]
#     print(deprel, '&', np.round(pearsonr(data_new['Is.Particular.Norm'], data_new['part.Pred'])[0] * 100, sigdig), '&', np.round(r1_score(data_new['Is.Particular.Norm'], data_new['part.Pred']) * 100, sigdig), '&', np.round(pearsonr(data_new['Is.Kind.Norm'], data_new['kind.Pred'])[0] * 100, sigdig), '&', np.round(r1_score(data_new['Is.Kind.Norm'], data_new['kind.Pred']) * 100, sigdig), '&', np.round(pearsonr(data_new['Is.Abstract.Norm'], data_new['abs.Pred'])[0] * 100, sigdig), '&', np.round(r1_score(data_new['Is.Abstract.Norm'], data_new['abs.Pred']) * 100, sigdig), '&', np.round(r1_score(data_new.loc[:, ['Is.Particular.Norm', 'Is.Kind.Norm', 'Is.Abstract.Norm']].values, data_new.loc[:, ['part.Pred', 'kind.Pred', 'abs.Pred']].values) * 100, sigdig), "\\\\")


# print("\npred POS")
# for pos in list(set(data_pred['POS'].tolist())):
#     data_new = data_pred[data_pred['POS'] == pos]
#     print(pos, '&', np.round(pearsonr(data_new['Is.Particular.Norm'], data_new['part.Pred'])[0] * 100, sigdig), '&', np.round(r1_score(data_new['Is.Particular.Norm'], data_new['part.Pred']) * 100, sigdig), '&', np.round(pearsonr(data_new['Is.Hypothetical.Norm'], data_new['hyp.Pred'])[0] * 100, sigdig), '&', np.round(r1_score(data_new['Is.Hypothetical.Norm'], data_new['hyp.Pred']) * 100, sigdig), '&', np.round(pearsonr(data_new['Is.Dynamic.Norm'], data_new['dyn.Pred'])[0] * 100, sigdig), '&', np.round(r1_score(data_new['Is.Dynamic.Norm'], data_new['dyn.Pred']) * 100, sigdig), '&', np.round(r1_score(data_new.loc[:, ['Is.Particular.Norm', 'Is.Hypothetical.Norm', 'Is.Dynamic.Norm']].values, data_new.loc[:, ['part.Pred', 'hyp.Pred', 'dyn.Pred']].values) * 100, sigdig), "\\\\")

# print("\npred DEPREL")
# # R1 and correlation based on POS and gov_rel
# for deprel in list(set(data_pred['DEPREL'].tolist())):
#     data_new = data_pred[data_pred['DEPREL'] == deprel]
#     print(deprel, '&', np.round(pearsonr(data_new['Is.Particular.Norm'], data_new['part.Pred'])[0] * 100, sigdig), '&', np.round(r1_score(data_new['Is.Particular.Norm'], data_new['part.Pred']) * 100, sigdig), '&', np.round(pearsonr(data_new['Is.Hypothetical.Norm'], data_new['hyp.Pred'])[0] * 100, sigdig), '&', np.round(r1_score(data_new['Is.Hypothetical.Norm'], data_new['hyp.Pred']) * 100, sigdig), '&', np.round(pearsonr(data_new['Is.Dynamic.Norm'], data_new['dyn.Pred'])[0] * 100, sigdig), '&', np.round(r1_score(data_new['Is.Dynamic.Norm'], data_new['dyn.Pred']) * 100, sigdig), '&', np.round(r1_score(data_new.loc[:, ['Is.Particular.Norm', 'Is.Hypothetical.Norm', 'Is.Dynamic.Norm']].values, data_new.loc[:, ['part.Pred', 'hyp.Pred', 'dyn.Pred']].values) * 100, sigdig), "\\\\")


# for pos in list(set(data_arg['POS'].tolist()).intersection()):
#     data_new = data_arg[data_arg['POS'] == pos]
#     data_new2 = data_pred[data_pred['POS'] == pos]
#     print(pos, '&', np.round(pearsonr(data_new['Is.Kind.Norm'], data_new['kind.Pred'])[0] * 100, sigdig), '&', np.round(r1_score(data_new['Is.Kind.Norm'], data_new['kind.Pred']) * 100, sigdig), '&', np.round(pearsonr(data_new['Is.Abstract.Norm'], data_new['abs.Pred'])[0] * 100, sigdig), '&', np.round(r1_score(data_new['Is.Abstract.Norm'], data_new['abs.Pred']) * 100, sigdig), '&', np.round(pearsonr(data_new2['Is.Hypothetical.Norm'], data_new2['hyp.Pred'])[0] * 100, sigdig), '&', np.round(r1_score(data_new2['Is.Hypothetical.Norm'], data_new2['hyp.Pred']) * 100, sigdig), "\\\\")

# for deprel in list(set(data_arg['DEPREL'].tolist())):
#     data_new = data_arg[data_arg['DEPREL'] == deprel]
#     data_new2 = data_pred[data_pred['DEPREL'] == deprel]
#     print(deprel, '&', np.round(pearsonr(data_new['Is.Kind.Norm'], data_new['kind.Pred'])[0] * 100, sigdig), '&', np.round(r1_score(data_new['Is.Kind.Norm'], data_new['kind.Pred']) * 100, sigdig), '&', np.round(pearsonr(data_new['Is.Abstract.Norm'], data_new['abs.Pred'])[0] * 100, sigdig), '&', np.round(r1_score(data_new['Is.Abstract.Norm'], data_new['abs.Pred']) * 100, sigdig), '&', np.round(pearsonr(data_new2['Is.Hypothetical.Norm'], data_new2['hyp.Pred'])[0] * 100, sigdig), '&', np.round(r1_score(data_new2['Is.Hypothetical.Norm'], data_new2['hyp.Pred']) * 100, sigdig), "\\\\")


# pron_df = data_arg[data_arg['Lemma'].isin(['you', 'they'])]
# print(pron_df[pron_df['Is.Kind.Norm'] > 0]['Sentences'])

# hyp_df = data_pred[(data_pred['Sentences'].str.contains('if'))]
print(data_pred[(data_pred['Sentences'].str.contains('if ', regex=False)) & (data_pred['hyp.Pred'] < -0.3)]['Sentences'])
