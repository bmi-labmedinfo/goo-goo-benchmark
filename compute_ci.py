import pandas as pd
import numpy as np


def compute_ci(perf_list, z=1.96):
    std = np.std(perf_list)
    mean = np.mean(perf_list)

    return (mean, z*std/np.sqrt(len(perf_list)))
def compute_ci_on_fraction(df, fraction=1, metric='F1'):

    df_f = df[df['fraction'] == fraction]
    model_names = df_f['model'].unique()
    model2ci = {}
    for model in model_names:
        df_f_model = df_f[df_f['model'] == model]
        (mean_model, interv_model) = compute_ci(df_f_model[metric].tolist())
        model2ci[model] = [round(mean_model-interv_model, 2), round(mean_model+interv_model, 2), round(mean_model, 2), round(np.std(df_f_model[metric].tolist()), 2)]
    return model2ci


def add_models(d, models):
    """
    In case for one set we miss a model (for example tabpfn 2 raw in 100%
    :param d:
    :param models:
    :return:
    """
    for m in models:
        if m not in d.keys():
            d[m] = ['-']*4
    return d


fpath = '/Users/giovannanicora/Downloads/cyrrosis_learning_curves_perfold.csv'
fpath = '/Users/giovannanicora/Downloads/Thyroid_cancer_learning_curves_perfold.csv'
fpath = '/Users/giovannanicora/Downloads/parkinson_learning_curves_perfold.csv'
fpath = '/Users/giovannanicora/Downloads/myocardial_infarction_learning_curves_perfold.csv'
fpath = '/Users/giovannanicora/Downloads/hepatitis_learning_curves_perfold.csv'
fpath = '/Users/giovannanicora/Downloads/glioma_learning_curves_perfold.csv'
fpath = '/Users/giovannanicora/Downloads/student_depression_learning_curves_perfold.csv'

df = pd.read_csv(fpath)

model2ci_01 = compute_ci_on_fraction(df, 0.1, metric='F1')
model2ci_02 = compute_ci_on_fraction(df, 0.2, metric='F1')
model2ci_05 = compute_ci_on_fraction(df, 0.5, metric='F1')
model2ci_75 = compute_ci_on_fraction(df, 0.75, metric='F1')
model2ci_1 = compute_ci_on_fraction(df, 1, metric='F1')

df_results = pd.DataFrame()
df_results['model'] = list(model2ci_01.keys())



model2ci_01 = add_models(model2ci_01, df_results['model'].tolist())
model2ci_02 = add_models(model2ci_02, df_results['model'].tolist())
model2ci_05 = add_models(model2ci_05, df_results['model'].tolist())
model2ci_75 = add_models(model2ci_75, df_results['model'].tolist())
model2ci_1 = add_models(model2ci_1, df_results['model'].tolist())


df_results['10%'] = [str((model2ci_01[x][2]))+'+-'+str((model2ci_01[x][3])) for x in df_results['model'].tolist()]
df_results['20%'] = [str((model2ci_02[x][2]))+'+-'+str((model2ci_02[x][3])) for x in df_results['model'].tolist()]
df_results['50%'] = [str((model2ci_05[x][2]))+'+-'+str((model2ci_05[x][3])) for x in df_results['model'].tolist()]
df_results['75%'] = [str((model2ci_75[x][2]))+'+-'+str((model2ci_75[x][3])) for x in df_results['model'].tolist()]
df_results['100%'] = [str((model2ci_1[x][2]))+'+-'+str((model2ci_1[x][3])) for x in df_results['model'].tolist()]

df_results.to_csv('depression_results.csv')


