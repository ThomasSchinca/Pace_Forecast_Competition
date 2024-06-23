# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 15:18:24 2024

@author: thoma
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt 
from shape import Shape,finder
from scipy.stats import norm
import properscoring as ps
from collections import Counter
np.random.seed(0)

# =============================================================================
# Functions 
# =============================================================================

def draw_samples(df, number,siz):
    results = []
    for col in df.columns:
        samples = np.random.choice(df[col], size=siz)
        for i, sample in enumerate(samples, start=1):
            results.append([number, col, i, sample])
    sampled_df = pd.DataFrame(results, columns=['country_id', 'month_id', 'draw', 'value'])
    return sampled_df

def draw_samples_from_mean(series, number,mode='pois'):
    results = []
    for j,val in enumerate(series):
        if mode =='pois':
            samples = np.random.poisson(val, 1000)
        elif mode =='exp':
            samples = np.random.exponential(val, 1000)
        for i, sample in enumerate(samples, start=1):
            results.append([number, series.index[j], i, sample])
    sampled_df = pd.DataFrame(results, columns=['country_id', 'month_id', 'draw', 'value'])
    return sampled_df

def calculate_crps(predictions, actual):
    N = len(predictions)
    crps = np.sum((np.sort(predictions) - actual) ** 2) / N
    return crps

bins = [0, 1, 3, 6, 11, 26, 51, 101, 251, 501, 1001]
bin_labels = range(len(bins) - 1)

def bin_predictions(predictions):
    binned = np.digitize(predictions, bins, right=True) - 1
    return binned

def calculate_ignorance_score(predictions, actual, bins, bin_labels):
    # Bin the predictions
    binned_predictions = bin_predictions(predictions)
    bin_counts = np.bincount(binned_predictions, minlength=len(bin_labels))
    bin_counts += 1
    probabilities = bin_counts / bin_counts.sum()
    actual_bin = np.digitize([actual], bins, right=True)[0] - 1
    ignorance_score = -np.log(probabilities[actual_bin])
    return ignorance_score


def calculate_mis_2(predictions, actual, alpha=0.05):
    lower = np.quantile(predictions, q = alpha/2, axis = -1)
    upper = np.quantile(predictions, q = 1 - (alpha/2), axis = -1)
    interval_width = upper - lower
    lower_coverage = (2/alpha)*(lower-actual) * (actual<lower)
    upper_coverage = (2/alpha)*(actual-upper) * (actual>upper)
    return(interval_width + lower_coverage + upper_coverage)

def cal_res(df_true,df_pred):
    results = []
    for _, row in df_true.iterrows():
        month_id = row['month_id']
        country_id = row['country_id']
        actual = row['outcome']
        predictions = df_pred[(df_pred['month_id'] == month_id) & 
                                     (df_pred['country_id'] == country_id)]['value'].values
        if len(predictions) == 0:
            continue
        crps = ps.crps_ensemble(actual, predictions)
        try:
            log_score = calculate_ignorance_score(predictions,actual,bins,bin_labels)
        except:
            log_score=0
        mis = mis = calculate_mis_2(predictions, actual, alpha=0.05)
        results.append({
            'CRPS': crps,
            'Log Score': log_score,
            'MIS': mis
        })
    results_df = pd.DataFrame(results)
    results_df = results_df.fillna(0)
    return results_df.mean()

# =============================================================================
# Data Import
# =============================================================================

df_country = pd.read_csv('Data/country_list.csv',index_col=0)
df_month = pd.read_csv('Data/month_ids.csv',index_col=0)
df_input = pd.read_parquet('Data/cm_features.parquet')
df_tot_m_tot = df_input.pivot(index='month_id', columns='country_id', values='ged_sb')

# =============================================================================
# Forecast Creation
# =============================================================================

res_sce=pd.DataFrame()
dict_res={i :[] for i in ['2018','2019','2020','2021','2022','2023','2024']}
for counter,annee in enumerate(['2018','2019','2020','2021','2022','2023']):
    df_tot_m = df_tot_m_tot.loc[:454+counter*12]
    df_tot_m=df_tot_m.fillna(0)
    h_train=10
    h=14
    pred_tot_b=pd.DataFrame()
    df_search = df_tot_m.loc[:, (df_tot_m != 0).any(axis=0)]
    for coun in range(len(df_tot_m.columns)):
        if not (df_tot_m.iloc[-h_train:,coun]==0).all():
            shape = Shape()
            shape.set_shape(df_tot_m.iloc[-h_train:,coun]) 
            find = finder(df_search.iloc[:-h,:],shape)
            find.find_patterns(min_d=0.5,select=True,metric='dtw',dtw_sel=2,min_mat=10,d_increase=0.05)
            clu=find.predict(horizon=14,plot=False,seq_out=True)
            clu_b = find.predict_best_sce(horizon=14,div_h=2.2)
            clu.columns = clu.columns + df_tot_m.iloc[-h_train:,coun].index[-1]+1
            clu_b.columns = clu_b.columns + df_tot_m.iloc[-h_train:,coun].index[-1]+1
            clu_b = clu_b*(df_tot_m.iloc[-h_train:,coun].max()-df_tot_m.iloc[-h_train:,coun].min())+df_tot_m.iloc[-h_train:,coun].min()
            clu_b[clu_b<0]=0
            pred_tot_b = pd.concat([pred_tot_b,draw_samples(clu_b,df_tot_m.iloc[-h_train:,coun].name,1000)])
        else :
            preds = pd.Series([0]*h,index=range(df_tot_m.iloc[-h_train:,coun].index[-1]+1,df_tot_m.iloc[-h_train:,coun].index[-1]+1+h))
            df_zer = draw_samples_from_mean(preds,df_tot_m.iloc[-h_train:,coun].name,mode='exp')
            pred_tot_b = pd.concat([pred_tot_b,df_zer])
            
    df_true = pd.read_parquet('Data/cm_actuals_'+annee+'.parquet')
    df_true=df_true.reset_index()
    res_sce = pd.concat([res_sce,cal_res(df_true,pred_tot_b)],axis=1)
    dict_res[annee]=pred_tot_b

df_tot_m = df_tot_m_tot
df_tot_m=df_tot_m.fillna(0)
h_train=10
h=14
pred_tot_b=pd.DataFrame()
df_search = df_tot_m.loc[:, (df_tot_m != 0).any(axis=0)]
for coun in range(len(df_tot_m.columns)):
    if not (df_tot_m.iloc[-h_train:,coun]==0).all():
        shape = Shape()
        shape.set_shape(df_tot_m.iloc[-h_train:,coun]) 
        find = finder(df_search.iloc[:-h,:],shape)
        find.find_patterns(min_d=0.5,select=True,metric='dtw',dtw_sel=2,min_mat=10,d_increase=0.05)
        clu=find.predict(horizon=14,plot=False,seq_out=True)
        clu_b = find.predict_best_sce(horizon=14,div_h=2.2)
        clu.columns = clu.columns + df_tot_m.iloc[-h_train:,coun].index[-1]+1
        clu_b.columns = clu_b.columns + df_tot_m.iloc[-h_train:,coun].index[-1]+1
        clu_b = clu_b*(df_tot_m.iloc[-h_train:,coun].max()-df_tot_m.iloc[-h_train:,coun].min())+df_tot_m.iloc[-h_train:,coun].min()
        clu_b[clu_b<0]=0
        pred_tot_b = pd.concat([pred_tot_b,draw_samples(clu_b,df_tot_m.iloc[-h_train:,coun].name,1000)])
    else :
        preds = pd.Series([0]*h,index=range(df_tot_m.iloc[-h_train:,coun].index[-1]+1,df_tot_m.iloc[-h_train:,coun].index[-1]+1+h))
        df_zer = draw_samples_from_mean(preds,df_tot_m.iloc[-h_train:,coun].name,mode='exp')
        pred_tot_b = pd.concat([pred_tot_b,df_zer])
dict_res['2024']=pred_tot_b

# =============================================================================
# Output creation
# =============================================================================

mont_lim=[457,469,481,493,505,517,535]
for counter,annee in enumerate(['2018','2019','2020','2021','2022','2023','2024']):
    df_out = dict_res[annee]
    df_out = df_out[df_out['month_id']>=mont_lim[counter]]
    df_out.to_parquet(f'Preds/window=Y{annee}/SF_{annee}.parquet')

# =============================================================================
# Comparison with Conflictology_12m
# =============================================================================

res_bench=pd.DataFrame()
for counter,annee in enumerate(['2018','2019','2020','2021','2022','2023']):
    bench = pd.read_parquet('Data/bm_'+annee+'.parquet')
    bench= bench.reset_index()
    bench = bench.iloc[:, [1, 0] + list(range(2, len(bench.columns)))]
    bench.columns = pred_tot_b.columns
    df_true = pd.read_parquet('Data/cm_actuals_'+annee+'.parquet')
    df_true=df_true.reset_index()
    res_bench = pd.concat([res_bench,cal_res(df_true,bench)],axis=1)

res_sce.columns=[*range(2018,2024)]
res_bench.columns=[*range(2018,2024)]
mean_sce = res_sce.mean(axis=1)
mean_bench = res_bench.mean(axis=1)
res_sce['Mean'] = mean_sce
res_bench['Mean'] = mean_bench
index = ['CRPS', 'Log Score', 'MIS']
columns = [*range(2018, 2024)]
columns.append('Mean')

fig, axes = plt.subplots(nrows=3, ncols=7, figsize=(24, 10))
fig.suptitle('Comparison of Metrics Across Datasets', fontsize=16)
axes = axes.flatten()
for i, metric in enumerate(index):
    for j, year in enumerate(columns):
        ax = axes[i*7 + j]
        values = [
            res_sce.loc[metric, year],
            res_bench.loc[metric, year],]
        ax.bar(['SCE', 'Bench'], values)
        if year == 2024:
            ax.set_title(f'{metric} - Mean')
        else:
            ax.set_title(f'{metric} - {year}')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(min(values)-0.1*min(values), max(values)+0.1*max(values))
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

