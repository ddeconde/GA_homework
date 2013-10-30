#!/usr/bin/env python

#
# NB: This file is a duplicate of hw4.py as the PCA capability has been
# "back-ported" into it.
#

from __future__ import division
from __future__ import print_function
import numpy as np
import csv
import sys
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV


# get_data() currently relies on csv.reader and expects a single input
# file formatted in a manner amenable to csv.reader digestion.
#
# This function might be extended in a number of ways using functions like
# numpy.loadtxt or sklearn.datasets.load_files or constructing an even
# more flexible data input module.

def get_data(infile_name):
    data = []
    with open(infile_name, 'rb') as data_source:
        reader = csv.reader(data_source)
        for row in reader:
            data.append(row)
    return scrub(data)


# In general this function should do whatever processing is necessary
# to return data in the numpy.array format that the sklearn functions
# require.
#
# For the time being this just "transposes" the argument to turn
# a list of pairs into a list of the first components and a list of
# the second components.
#
# The input file format this function currently expects is a CSV file
# where each row is of the form:
#     predictor, response

def scrub(data):
    data_t = zip(*data)
    return np.array(data_t[0]), np.array(data_t[1])


# Save a plot directly to a .png file rather than displaying it. Useful
# for terminal based operation.

def gen_graphic_file(outfile, predictors):
    plot = plot_graph(predictors)
    plt.savefig(outfile, bbox_inches='tight')


# Use matplotlib tools to generate a plot of explained variance by
# principal component

def plot_graph(predictors):
    pca = PCA()
    pca.fit(predictors)

    fig, ax = plt.subplots(1)
    ax.plot(pca.explained_variance_, lw=2, label="explained variance",
            color="blue")
    ax.set_title("variance by component")
    ax.legend(loc="best")
    ax.set_xlabel('principal components')
    ax.set_ylabel('explained variance')
    ax.grid()
    return fig


def main():
    # Parse arguments when run from CLI

    parser = argparse.ArgumentParser()
    parser.add_argument('infile', nargs='?', default="iris")
    parser.add_argument("-g", "--graph", nargs='?', const=".png",
            default="none")
    parser.add_argument("-p", "--pca", type=int, nargs='?', const=0,
            help="specify a minimum number of neighbors")
    parser.add_argument("-k", "--kfold", type=int, default=5,
            help="specify the number of partitions for cross validation")

    args = parser.parse_args()


    # Acquire data

    if args.infile != "iris":
        # when input file is given, pull data from there
        predictors, responses = get_data(args.infile)
    else:
        # when no input is given, default to using iris data
        iris_data = datasets.load_digits()
        predictors, responses = iris_data.data, iris_data.target

    # hyperparameter lists for grid search
    components = [20, 30, 40, 50]
    cs = np.logspace(-5, -1, 3)

    clf = LogisticRegression()

    if args.pca:
        if args.pca is not 0:
            components = [args.pca]
        pca = PCA()
        pipe = Pipeline(steps=[('pca', pca), ('clf', clf)])
        params = dict(pca__n_components=components, clf__C=cs)
    else:
        pipe = Pipeline(steps=[('clf', clf)])
        params = dict(clf__C=cs)

    print("Fitting model...")
    estimator = GridSearchCV(pipe, params, cv=args.kfold)
    estimator.fit(predictors, responses)
    score = estimator.best_score_


    # Output results

    if args.pca:
        model = " with {} principal components ".format(
                estimator.best_params_[n_components])
    else:
        model = " "
    print("Logistic regression{}score: {}".format(model, score))


    # Produce graphs

    if args.graph is not "none":
        if args.graph == ".png":
            gen_graphic_file(args.infile + args.graph, predictors)
        else:
            gen_graphic_file(arga.graphic + ".png", predictors)


if __name__ == '__main__':
    main()

