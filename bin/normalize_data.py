import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import rpy2
import matplotlib.pyplot as plt
import seaborn as sns
from factslab.utility import ridit

if __name__ == __main__:

    data_arg = pd.read_csv('arg_raw_data.tsv', sep="\t").dropna()
    data_prd = pd.read_csv('pred_raw_data.tsv', sep="\t").dropna()

    data_arg['Unique.ID'] = data_arg.apply(lambda x: x['Split'] + " sent_" + str(x['Sentence.ID']) + "_" + str(x["Arg.Span"]), axis=1)
    data_prd['Unique.ID'] = data_prd.apply(lambda x: x['Split'] + " sent_" + str(x['Sentence.ID']) + "_" + str(x["Pred.Span"]), axis=1)

    data_arg['Is.Particular.Confidence.Norm'] = data_arg.groupby(['Split', 'Annotator.ID'])['Is.Particular.Confidence'].transform(ridit)
    data_arg['Is.Kind.Confidence.Norm'] = data_arg.groupby(['Split', 'Annotator.ID'])['Is.Kind.Confidence'].transform(ridit)
    data_arg['Is.Abstract.Confidence.Norm'] = data_arg.groupby(['Split', 'Annotator.ID'])['Is.Abstract.Confidence'].transform(ridit)

    data_prd['Is.Particular.Confidence.Norm'] = data_prd.groupby(['Split', 'Annotator.ID'])['Is.Particular.Confidence'].transform(ridit)
    data_prd['Is.Hypothetical.Confidence.Norm'] = data_prd.groupby(['Split', 'Annotator.ID'])['Is.Hypothetical.Confidence'].transform(ridit)
    data_prd['Is.Dynamic.Confidence.Norm'] = data_prd.groupby(['Split', 'Annotator.ID'])['Is.Dynamic.Confidence'].transform(ridit)

