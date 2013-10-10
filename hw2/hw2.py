#!/usr/bin/env python

from __future__ import division
from __future__ import print_function
import numpy as np
import csv
import sys
import argparse
from sklearn import datasets
from sklearn import cross_validation
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
import warnings


def get_data(infile_name):
    data = []
    with open(infile_name, 'rb') as data_source:
        reader = csv.reader(data_source)
        for row in reader:
            data.append(row)
    return scrub(data)


def scrub(data):
    data_t = zip(*data)
    return np.array(data_t[0]), np.array(data_t[1])


### main ###

def main():
    # Parse arguments when run from CLI

    parser = argparse.ArgumentParser()
    parser.add_argument('infile', nargs='?', default="diabetes")
    parser.add_argument("-m", "--models", nargs='*', default=["ols",],
        choices=["ols", "ridge", "lasso", "enet"],
        help= "specify which linear model(s) to use: ols, ridge, lasso, enet.")
    parser.add_argument("-k", "--kfold", type=int, default=5, help=
        "specify the number of partitions for cross validation")

    args = parser.parse_args()


    # Acquire data

    if args.infile != "diabetes":
        # when input file is given, pull data from there
        predictors, responses = get_data(args.infile)
    else:
        # when no input is given, default to using diabetes data
        diabetes = datasets.load_diabetes()
        predictors, responses = diabetes.data, diabetes.target

    scores = {}
    n_alphas = 200
    alphas = np.logspace(-10, 0, n_alphas)
    l1_ratios = [.1, .5, .7, .9, .95, .99, 1]

    for model in args.models:
        # no switch statement in Python so...
        if model == "ridge":
            lm = Ridge(fit_intercept=False)
            params = dict(lm__alpha=alphas)
        elif model == "lasso":
            lm = Lasso(fit_intercept=False, max_iter=4000)
            params = dict(lm__alpha=alphas)
        elif model == "enet":
            lm = ElasticNet(fit_intercept=False, max_iter=4000)
            params = dict(lm__alpha=alphas, lm__l1_ratio=l1_ratios)
        else:
            lm = LinearRegression()
            params = {}

        scaler = StandardScaler()
        pipe = Pipeline(steps=[('scale', scaler), ('lm', lm)])
        #decomp = PCA()
        #pipe = Pipeline(steps=[('decomp', decomp), ('lm', lm)])

        print("Fitting {} linear model. This may take a moment...".format(
            model))
        # the coordinate descent procedure used by the lasso and elastic
        # net models to calculate the fit yields a warning about
        # convergence, but given enough iterations it produces a result
        # so this warning is suppressed via the catch_warnings context
        #
        # this is a less than satisfying solution
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimator = GridSearchCV(pipe, params, cv=args.kfold)
            estimator.fit(predictors, responses)
            scores[model] = estimator.best_score_

        #print(lm.coef_)
        #print(lm.intercept_)

    for model in args.models:
        print("The R^2 score of the {} model was: {}".format(model,
            scores[model]))


if __name__ == '__main__':
    main()


