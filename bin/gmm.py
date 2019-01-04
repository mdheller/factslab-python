import pandas as pd
import numpy as np
from os.path import expanduser
from scipy.stats import spearmanr, pearsonr
from sklearn import mixture
import itertools
from scipy import linalg
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from collections import defaultdict

np.random.seed(0)
if __name__ == "__main__":
    home = expanduser('~')

    datafile = home + "/Desktop/protocols/data/noun_raw_data_norm_122218.tsv"
    arg_raw = pd.read_csv(datafile, sep="\t")
    arg_raw = arg_raw[arg_raw['Split'].isin(['train', 'dev'])]
    arg_raw['Sentence.ID'] = arg_raw['Sentence.ID'].str.replace('sent_', '', regex=False)
    arg_raw['Unique.ID'] = arg_raw.apply(lambda x: str(x['Sentence.ID']) + "_" + str(x['Span']).split(',')[0] + "_" + str(x['Span']).split(',')[-1], axis=1)
    arg = arg_raw.dropna()
    arg = arg.groupby('Unique.ID', as_index=False).mean().reset_index(drop=True)
    sid_dict = arg_raw[['Unique.ID', 'Sentence.ID', 'Context.Span']].set_index('Unique.ID').to_dict('index')
    arg['Sentence.ID'] = arg['Unique.ID'].apply(lambda x: sid_dict[x]['Sentence.ID'])
    arg['Context.Span'] = arg['Unique.ID'].apply(lambda x: sid_dict[x]['Context.Span'])
    arg = arg.rename(columns={'Is.Particular.Norm': 'Is.Particular.Arg.Norm'})
    arg['pred.ID'] = arg.apply(lambda x: str(x['Sentence.ID']) + "_" + str(x['Context.Span']), axis=1)

    datafile = home + "/Desktop/protocols/data/pred_raw_data_norm_122218.tsv"
    pred_raw = pd.read_csv(datafile, sep="\t")
    pred_raw = pred_raw[pred_raw['Split'].isin(['train', 'dev'])]
    pred_raw['Sentence.ID'] = pred_raw['Sentence.ID'].str.replace('sent_', '', regex=False)
    pred_raw['Unique.ID'] = pred_raw.apply(lambda x: str(x['Sentence.ID']) + "_" + str(x["Span"]), axis=1)
    pred = pred_raw.groupby('Unique.ID', as_index=False).mean().reset_index(drop=True)
    pred = pred.rename(columns={'Is.Particular.Norm': 'Is.Particular.Pred.Norm'})

    pred_cols = ['Is.Particular.Pred.Norm', 'Is.Dynamic.Norm', 'Is.Hypothetical.Norm']
    arg_cols = ['Is.Particular.Arg.Norm', 'Is.Kind.Norm', 'Is.Abstract.Norm']

    df = arg
    pred_dict = pred[['Unique.ID', 'Is.Particular.Pred.Norm', 'Is.Dynamic.Norm', 'Is.Hypothetical.Norm']].set_index('Unique.ID').to_dict('index')
    pred_dict = defaultdict(lambda: defaultdict(None), pred_dict)
    for col in pred_cols:
        df[col] = df['pred.ID'].apply(lambda x: pred_dict[x].get(col))

df = df.dropna()
X = df.loc[:, pred_cols + arg_cols].values
lowest_aic = np.infty
aic = []
n_components_range = range(1, 11)
cv_types = ['spherical', 'tied', 'diag', 'full']
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type)
        gmm.fit(X)
        aic.append(gmm.aic(X))
        if aic[-1] < lowest_aic:
            lowest_aic = aic[-1]
            best_gmm = gmm

aic = np.array(aic)
color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                              'darkorange'])
clf = best_gmm
bars = []

# Plot the aic scores
plt.figure(figsize=(8, 6))
spl = plt.subplot(2, 1, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + .2 * (i - 2)
    bars.append(plt.bar(xpos, aic[i * len(n_components_range):
                                  (i + 1) * len(n_components_range)],
                        width=.2, color=color))
plt.xticks(n_components_range)
plt.ylim([aic.min() * 1.01 - .01 * aic.max(), aic.max()])
plt.title('aic score per model')
xpos = np.mod(aic.argmin(), len(n_components_range)) + .65 +\
    .2 * np.floor(aic.argmin() / len(n_components_range))
plt.text(xpos, aic.min() * 0.97 + .03 * aic.max(), '*', fontsize=14)
spl.set_xlabel('Number of components')
spl.legend([b[0] for b in bars], cv_types)

# Plot the winner
splot = plt.subplot(2, 1, 2)
Y_ = clf.predict(X)
for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,
                                           color_iter)):
    v, w = linalg.eigh(cov)
    if not np.any(Y_ == i):
        continue
    plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

    # Plot an ellipse to show the Gaussian component
    angle = np.arctan2(w[0][1], w[0][0])
    angle = 180. * angle / np.pi  # convert to degrees
    v = 2. * np.sqrt(2.) * np.sqrt(v)
    ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
    ell.set_clip_box(splot.bbox)
    ell.set_alpha(.5)
    splot.add_artist(ell)

plt.xticks(())
plt.yticks(())
plt.title('Selected GMM: full model ' + str(len(clf.means_)) + ' components')
plt.subplots_adjust(hspace=.35, bottom=.02)
plt.savefig('gmm.png')
