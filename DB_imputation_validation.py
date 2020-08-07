#!/usr/bin/env python

from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor

@ignore_warnings(category=ConvergenceWarning)
def fill_missing(model, pim_df,log_cols, name):
    log_perms = pim_df[log_cols].to_numpy()
    log_cp = log_perms.copy()
    transformer = model
    log_full = transformer.fit_transform(log_cp)
    new_cols = [log_gas+'_'+name for log_gas in log_cols]
    new_df = pd.DataFrame(data=log_full, columns=new_cols)
    return new_df


def validation(train_df, target_df, log_gases, model, name):
    train_size = len(train_df)
    log_gases_Bayesian = [l +'_'+name for l in log_gases]
    df1 = pd.DataFrame()
    df2 = target_df 
    for col in log_gases:
        df1[col] = train_df[col + '_'+ name]
    df_to_fill = pd.concat([df1, df2],sort=False)
    df_to_fill.to_csv('tmp.csv',index=False)
    df_to_fill = pd.read_csv('tmp.csv')
    df_filled = fill_missing(model, df_to_fill, log_gases, name)
    dd = df_filled.iloc[train_size:]
    #dd.loc[:,'idx'] = list(range(len(target_df)))
    #target_df.loc[:,'idx'] = list(range(len(target_df)))
    dd.reset_index(inplace=True)
    dd = dd.drop(['index'],axis=1)
    res_df = pd.concat([target_df, dd],axis=1) 
    return res_df

def main():
    pass

if __name__ == '__main__':
    log_gases = ['log10_He','log10_H2', 'log10_O2', 'log10_N2', 'log10_CO2', 'log10_CH4']
    model2 = IterativeImputer(random_state=0,max_iter=200)
    train_df = pd.read_csv('stepwise_fill/Aus_Bayesian_filled_200.csv',encoding='ISO-8859-1')
    target_df = pd.read_csv('stepwise_fill/combined_val_raw.csv')
    #model3 =IterativeImputer(estimator= LinearRegression(), random_state=0, max_iter=999)
    #model4 = IterativeImputer(estimator=DecisionTreeRegressor(), random_state=0, tol=0.01,max_iter=999)
    #model5 = IterativeImputer(estimator=RandomForestRegressor(n_estimators=20), random_state=0, max_iter=1000)
    #model6 = IterativeImputer(estimator=XGBRegressor(),random_state=0,max_iter=1000)
    model7 = IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=100),random_state=0,max_iter=200)
    #model8 = IterativeImputer(estimator=KNeighborsRegressor(n_neighbors=15),random_state=0,max_iter=1000)
    #iter_validation(dir='stepwise_fill/',fname='combined_val_raw',log_gases=log_gases, model=model2,name='Bayesian' )
    res_df = validation(train_df, target_df, log_gases, model2, 'Bayesian')
    res_df.to_csv('test_res.csv',index=False)
