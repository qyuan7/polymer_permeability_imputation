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
import logging

@ignore_warnings(category=ConvergenceWarning)
def fill_missing(model, pim_df,log_cols, name):
    log_perms = pim_df[log_cols].to_numpy()
    log_cp = log_perms.copy()
    transformer = model
    log_full = transformer.fit_transform(log_cp)
    new_cols = [log_gas+'_'+name for log_gas in log_cols]
    new_df = pd.DataFrame(data=log_full, columns=new_cols)
    r_df = pd.concat([pim_df, new_df], axis=1)
    return r_df


def main():
    log_gases = ['log10_He', 'log10_H2', 'log10_O2', 'log10_N2', 'log10_CO2', 'log10_CH4']
    parser = argparse.ArgumentParser(description="Database imputation for Gas Separation Polymers")
    parser.add_argument('database_path')
    parser.add_argument('--model', action='store', dest='model',choices=['Bayesian','etree'],
                        default='Bayesian', help='What model to use: Bayesian; etree')
    args = parser.parse_args()
    dfname, m = args.database_path, args.model
    if m == 'Bayesian':
        model =  IterativeImputer(random_state=0,max_iter=200)
    else:
        model = IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=100),
                                 random_state=0,
                                 max_iter=200)
    df_to_fill = pd.read_csv(dfname)
    df_imputed = fill_missing(model, df_to_fill, log_gases, m)
    fname = dfname[:-4]
    df_imputed.to_csv(fname+f'_{m}.csv',index=False)


if __name__ == '__main__':
    main()
