#!/usr/bin/env python


from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
import argparse


@ignore_warnings(category=ConvergenceWarning)
def fill_missing(model, pim_df,log_cols, name):
    log_perms = pim_df[log_cols].to_numpy()
    log_cp = log_perms.copy()
    transformer = model
    log_full = transformer.fit_transform(log_cp)
    new_cols = [log_gas+'_'+name for log_gas in log_cols]
    new_df = pd.DataFrame(data=log_full, columns=new_cols)
    return new_df


def main():
    pass
if __name__ == '__main__':
    log_gases = ['log10_He', 'log10_H2', 'log10_O2', 'log10_N2', 'log10_CO2', 'log10_CH4']
    model2 = IterativeImputer(random_state=0,max_iter=10)
    model3 = IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=100),random_state=99,max_iter=1000)
    df1 = pd.read_csv('Aus_to_fill_clear.csv')
    #dfnorm = fill_missing(model1, df1, log_gases, 'Norm')
    #dfbayesian = fill_missing(model2, df1, log_gases, 'Bayesian')
    dfetree = fill_missing(model3, df1, log_gases, 'etree')
    df = pd.concat([df1,  dfetree], axis=1)
    df.to_csv('stepwise_fill/Aus_etree_filled_1000.csv',index=False)
