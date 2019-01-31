import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import rpy2
import matplotlib.pyplot as plt
import seaborn as sns
from factslab.utility import ridit

def fit_glmem_normalizer(X, annid, uniqueid, weights, seed, loss="binomial", iterations=20000):
    np.random.seed(seed)

    if loss == "binomial":
        X_tf = tf.constant(X.astype(np.float64))
    elif loss in ["hinge", "l1"]:
        X_tf = tf.constant((2.*X.astype(np.float64)-1.).astype(np.float64))

    weights_tf = tf.constant(weights.astype(np.float64))

    fixed = tf.Variable(np.random.normal(size=X.shape[1]).astype(np.float64))

    nann = np.unique(annid).shape[0]
    nunq = np.unique(uniqueid).shape[0]

    random_ann = tf.Variable(np.random.normal(size=[nann, X.shape[1]]).astype(np.float64))
    random_ann_mean = tf.reduce_mean(random_ann, axis=0)[None,:]
    random_ann_centered = random_ann - random_ann_mean
    random_ann_standardized = random_ann_centered/tf.sqrt(tf.reduce_mean(tf.square(random_ann_centered), axis=0))[None,:]

    random_unq = tf.Variable(np.random.normal(size=[nunq, X.shape[1]]).astype(np.float64))
    #random_unq_mean = tf.reduce_mean(random_unq, axis=0)[None,:]
    #random_unq_centered = random_unq - random_unq_mean
    #random_unq_standardized = random_unq_centered/tf.sqrt(tf.reduce_mean(tf.square(random_unq_centered), axis=0))[None,:]

    logits = fixed[None,:] + tf.gather(random_ann_centered, annid) + tf.gather(random_unq, uniqueid)

    if X.shape[1] > 1:
        random_ann_covar = tf.reduce_mean(random_ann_standardized[:,:,None] * random_ann_standardized[:,None,:], axis=0)
    #     ann_dist = tfp.distributions.MultivariateNormalFullCovariance(loc=tf.zeros(X.shape[1], dtype=tf.float64),
    #                                                 covariance_matrix=random_ann_covar)
        ann_dist = tfp.distributions.MultivariateNormalFullCovariance(loc=tf.zeros(X.shape[1], dtype=tf.float64), covariance_matrix=random_ann_covar*tf.eye(X.shape[1], dtype=tf.float64))

    #hyperprior = tfp.distributions.LKJ(np.int64(3), np.float64(10.))

    nobs = np.prod(X.shape)
    nobsweighted = np.sum(weights)

    if loss == "binomial":
        loss_data = -tf.reduce_sum(weights_tf*(X_tf*tf.log(prob) + (1.-X_tf)*tf.log(1.-prob)))*nobs/nobsweighted
    elif loss == "hinge":
#         loss_data = tf.reduce_sum(weights_tf*tf.maximum(tf.zeros(X.shape, dtype=tf.float64), 1.-X_tf*logits))*nobs/nobsweighted
        loss_data = tf.reduce_sum(tf.maximum(tf.zeros(X.shape, dtype=tf.float64), weights_tf-X_tf*logits))
    elif loss == "l1":
        loss_data = tf.reduce_sum(tf.abs(weights_tf*X_tf-tf.tanh(logits)))

#     loss = loss_data -\
#            tf.reduce_sum(ann_dist.log_prob(random_ann_standardized)) -\
#            hyperprior.log_prob(random_ann_covar)

    if X.shape[1] > 1:
        loss = loss_data -\
               tf.reduce_sum(ann_dist.log_prob(random_ann_standardized))
    else:
        loss = loss_data -\
               tf.reduce_sum(tf.square(random_ann_standardized))

    optimizer = tf.train.AdamOptimizer(.01)

    train_op = optimizer.minimize(loss)

    sess.run(tf.global_variables_initializer())

    prev_best = np.inf

    for i in range(iterations):
        _, l, ld = sess.run([train_op, loss, loss_data])
        if l < prev_best:
            prev_best = l
            best_params = (fixed.eval(), random_ann_centered.eval(), random_unq.eval())
        if not i % 100:
#             print(i, l/nobs, ld/nobs)

    return best_params


if __name__ == __main__:

    data_arg = pd.read_csv('arg_long_data.tsv', sep="\t").dropna()
    data_prd = pd.read_csv('pred_long_data.tsv', sep="\t").dropna()

    data_arg['Unique.ID'] = data_arg.apply(lambda x: x['Split'] + " sent_" + str(x['Sentence.ID']) + "_" + str(x["Arg.Span"]), axis=1)
    data_prd['Unique.ID'] = data_prd.apply(lambda x: x['Split'] + " sent_" + str(x['Sentence.ID']) + "_" + str(x["Pred.Span"]), axis=1)

    data_arg['Is.Particular.Confidence.Norm'] = data_arg.groupby(['Split', 'Annotator.ID'])['Is.Particular.Confidence'].transform(ridit)
    data_arg['Is.Kind.Confidence.Norm'] = data_arg.groupby(['Split', 'Annotator.ID'])['Is.Kind.Confidence'].transform(ridit)
    data_arg['Is.Abstract.Confidence.Norm'] = data_arg.groupby(['Split', 'Annotator.ID'])['Is.Abstract.Confidence'].transform(ridit)

    data_prd['Is.Particular.Confidence.Norm'] = data_prd.groupby(['Split', 'Annotator.ID'])['Is.Particular.Confidence'].transform(ridit)
    data_prd['Is.Hypothetical.Confidence.Norm'] = data_prd.groupby(['Split', 'Annotator.ID'])['Is.Hypothetical.Confidence'].transform(ridit)
    data_prd['Is.Dynamic.Confidence.Norm'] = data_prd.groupby(['Split', 'Annotator.ID'])['Is.Dynamic.Confidence'].transform(ridit)

    data_arg_traindev = data_arg[data_arg.Split.isin(['train', 'dev'])]

    data_arg_traindev['Annotator.ID'] = data_arg_traindev['Annotator.ID'].astype('category')
    data_arg_traindev['annid'] = data_arg_traindev['Annotator.ID'].cat.codes


    data_arg_traindev['Unique.ID'] = data_arg_traindev['Unique.ID'].astype('category')
    data_arg_traindev['uniqueid'] = data_arg_traindev['Unique.ID'].cat.codes
    X_arg_traindev = data_arg_traindev[['Is.Particular', 'Is.Kind', 'Is.Abstract']].values
    annid_arg_traindev = data_arg_traindev['annid'].values.astype(np.int32)
    uniqueid_arg_traindev = data_arg_traindev['uniqueid'].values.astype(np.int32)
    weights_arg_traindev = data_arg_traindev[['Part.Confidence.Norm', 'Kind.Confidence.Norm', 'Abs.Confidence.Norm']].values

    params_arg_traindev = fit_glmem_normalizer(X_arg_traindev,
                                               annid_arg_traindev,
                                               #hitid_arg_traindev,
                                               uniqueid_arg_traindev,
                                               weights_arg_traindev,
                                               seed=29304926,
                                              loss="hinge",
                                              iterations=50000)

    fixed_arg_traindev, random_ann_arg_traindev, random_unq_arg_traindev = params_arg_traindev

    norm_arg_traindev = pd.DataFrame(fixed_arg_traindev[None,:] + random_unq_arg_traindev[data_arg_traindev['Unique.ID'].cat.codes], columns=['Is.Particular.Norm', 'Is.Kind.Norm', 'Is.Abstract.Norm'])

    lemma_arg_traindev = norm_arg_traindev.copy()
    lemma_arg_traindev['Lemma'] = data_arg_traindev['Lemma'].str.lower().values
    lemma_arg_traindev = lemma_arg_traindev.groupby('Lemma')[['Is.Particular.Norm', 'Is.Kind.Norm', 'Is.Abstract.Norm']].mean()
    lemma_arg_traindev = lemma_arg_traindev.reset_index()

    norm_arg_traindev['Unique.ID'] = data_arg_traindev['Unique.ID'].values
    norm_arg_traindev = norm_arg_traindev.groupby('Unique.ID')[['Is.Particular.Norm', 'Is.Kind.Norm', 'Is.Abstract.Norm']].mean()

    data_arg_test = data_arg[data_arg.Split.isin(['test'])]

    data_arg_test['Annotator.ID'] = data_arg_test['Annotator.ID'].astype('category')
    data_arg_test['annid'] = data_arg_test['Annotator.ID'].cat.codes

    data_arg_test['Unique.ID'] = data_arg_test['Unique.ID'].astype('category')
    data_arg_test['uniqueid'] = data_arg_test['Unique.ID'].cat.codes

    X_arg_test = data_arg_test[['Is.Particular', 'Is.Kind', 'Is.Abstract']].values
    annid_arg_test = data_arg_test['annid'].values.astype(np.int32)
    uniqueid_arg_test = data_arg_test['uniqueid'].values.astype(np.int32)
    weights_arg_test = data_arg_test[['Part.Confidence.Norm', 'Kind.Confidence.Norm', 'Abs.Confidence.Norm']].values

    params_arg_test = fit_glmem_normalizer(X_arg_test,
                                           annid_arg_test,
                                           uniqueid_arg_test,
                                           weights_arg_test,
                                           seed=29304927,
                                           loss="hinge",
                                           iterations=50000)

    fixed_arg_test, random_ann_arg_test, random_unq_arg_test = params_arg_test

    norm_arg_test = pd.DataFrame(fixed_arg_test[None,:] + random_unq_arg_test[data_arg_test['Unique.ID'].cat.codes], columns=['Is.Particular.Norm', 'Is.Kind.Norm', 'Is.Abstract.Norm'])

    norm_arg_test['Unique.ID'] = data_arg_test['Unique.ID'].values

    norm_arg_test = norm_arg_test.groupby('Unique.ID')[['Is.Particular.Norm', 'Is.Kind.Norm', 'Is.Abstract.Norm']].mean()

    data_prd_traindev = data_prd[data_prd.Split.isin(['train', 'dev'])]
    data_prd_traindev['Annotator.ID'] = data_prd_traindev['Annotator.ID'].astype('category')
    data_prd_traindev['annid'] = data_prd_traindev['Annotator.ID'].cat.codes

    data_prd_traindev['HIT.ID'] = data_prd_traindev['HIT.ID'].astype('category')
    data_prd_traindev['hitid'] = data_prd_traindev['HIT.ID'].cat.codes

    data_prd_traindev['Unique.ID'] = data_prd_traindev['Unique.ID'].astype('category')
    data_prd_traindev['uniqueid'] = data_prd_traindev['Unique.ID'].cat.codes

    X_prd_traindev = data_prd_traindev[['Is.Particular', 'Is.Dynamic', 'Is.Hypothetical']].values
    annid_prd_traindev = data_prd_traindev['annid'].values.astype(np.int32)
    hitid_prd_traindev = data_prd_traindev['hitid'].values.astype(np.int32)
    uniqueid_prd_traindev = data_prd_traindev['uniqueid'].values.astype(np.int32)
    weights_prd_traindev = data_prd_traindev[['Part.Confidence.Norm', 'Dyn.Confidence.Norm', 'Hyp.Confidence.Norm']].values

    params_prd_traindev = fit_glmem_normalizer(X_prd_traindev,
                                               annid_prd_traindev,
                                               uniqueid_prd_traindev,
                                               weights_prd_traindev,
                                               seed=4039281,
                                              loss="hinge",
                                              iterations=50000)

    fixed_prd_traindev, random_ann_prd_traindev, random_unq_prd_traindev = params_prd_traindev

    norm_prd_traindev = pd.DataFrame(fixed_prd_traindev[None,:] + random_unq_prd_traindev[data_prd_traindev['Unique.ID'].cat.codes], columns=['Is.Particular.Norm', 'Is.Dynamic.Norm', 'Is.Hypothetical.Norm'])

    lemma_prd_traindev = norm_prd_traindev.copy()
    lemma_prd_traindev['Lemma'] = data_prd_traindev['Lemma'].str.lower().values
    lemma_prd_traindev = lemma_prd_traindev.groupby('Lemma')[['Is.Particular.Norm', 'Is.Dynamic.Norm', 'Is.Hypothetical.Norm']].mean()
    lemma_prd_traindev = lemma_prd_traindev.reset_index()

    norm_prd_traindev['Unique.ID'] = data_prd_traindev['Unique.ID'].values

    norm_prd_traindev = norm_prd_traindev.groupby('Unique.ID')[['Is.Particular.Norm', 'Is.Dynamic.Norm', 'Is.Hypothetical.Norm']].mean()

    data_prd_test = data_prd[data_prd.Split.isin(['test'])]

    data_prd_test['Annotator.ID'] = data_prd_test['Annotator.ID'].astype('category')
    data_prd_test['annid'] = data_prd_test['Annotator.ID'].cat.codes
    data_prd_test['Unique.ID'] = data_prd_test['Unique.ID'].astype('category')
    data_prd_test['uniqueid'] = data_prd_test['Unique.ID'].cat.codes

    X_prd_test = data_prd_test[['Is.Particular', 'Is.Dynamic', 'Is.Hypothetical']].values
    annid_prd_test = data_prd_test['annid'].values.astype(np.int32)
    uniqueid_prd_test = data_prd_test['uniqueid'].values.astype(np.int32)
    weights_prd_test = data_prd_test[['Part.Confidence.Norm', 'Dyn.Confidence.Norm', 'Hyp.Confidence.Norm']].values

    params_prd_test = fit_glmem_normalizer(X_prd_test,
                                           annid_prd_test,
                                           uniqueid_prd_test,
                                           weights_prd_test,
                                           seed=4039281,
                                           loss="hinge",
                                           iterations=50000)

    fixed_prd_test, random_ann_prd_test, random_unq_prd_test = params_prd_test

    norm_prd_test = pd.DataFrame(fixed_prd_test[None,:] + random_unq_prd_test[data_prd_test['Unique.ID'].cat.codes], columns=['Is.Particular.Norm', 'Is.Dynamic.Norm', 'Is.Hypothetical.Norm'])

    norm_prd_test['Unique.ID'] = data_prd_test['Unique.ID'].values

    norm_prd_test = norm_prd_test.groupby('Unique.ID')[['Is.Particular.Norm', 'Is.Dynamic.Norm', 'Is.Hypothetical.Norm']].mean()

    # Merging traindev and test, apply threshold(maybe), and write to file
    norm_arg = pd.concat([norm_arg_traindev.reset_index(), norm_arg_test.reset_index()]).set_index('Unique.ID')
    norm_prd = pd.concat([norm_prd_traindev.reset_index(), norm_prd_test.reset_index()]).set_index('Unique.ID')

    data_arg_norm = pd.merge(data_arg, norm_arg, left_on='Unique.ID', right_index=True)
    data_prd_norm = pd.merge(data_prd, norm_prd, left_on='Unique.ID', right_index=True)

    data_arg_norm_thresh = pd.merge(data_arg_norm, np.minimum(1, np.maximum(-1, norm_arg.rename(columns={'Is.Particular.Norm': 'Is.Particular.Norm.Thresholded', left_on='Unique.ID', right_index=True)
    data_prd_norm_thresh = pd.merge(data_prd_norm, np.minimum(1, np.maximum(-1, norm_prd.rename(columns={'Is.Particular.Norm': 'Is.Particular.Norm.Thresholded', left_on='Unique.ID', right_index=True)

    data_arg_norm_thresh.to_csv('noun_raw_data_norm.tsv', sep='\t', index=False)
    data_prd_norm_thresh.to_csv('pred_raw_data_norm.tsv', sep='\t', index=False)