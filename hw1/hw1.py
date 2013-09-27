#!/usr/bin/env python

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
from sklearn.neighbors import KNeighborsClassifier


# A convenience function to wrap up the iteration that cross validates
# knn fitting over a range of neighbor counts and yields a list of means
# and a list of standard deviations.

def cross_val_over_neighbors(predictors, reponses, max_neighbors,
        min_neighbors, k_fold):
    neighbors = range(min_neighbors, max_neighbors + 1)
    score_means = []
    score_stds = []

    for n in neighbors:
        knn = KNeighborsClassifier(n)
        scores = cross_validation.cross_val_score(knn, predictors,
                responses, cv=k_fold)
        score_means.append(scores.mean())
        score_stds.append(scores.std())

    return neighbors, score_means, score_stds


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


# Basic text output defaults to stdout but can be directed to an output
# file using the "-o" argument.
#
# While the current form works well enough for dumping to CSV files,
# it is somewhat ugly when viewed in stdout.

def put_results(outfile_name, results):
    header = ["neighbors", "accuracy", "std_dev"]

    writer = csv.writer(outfile_name)
    writer.writerow(header)
    for row in results:
        writer.writerow(row)


# Save a plot directly to a .png file rather than displaying it. Useful
# for terminal based operation.

def gen_graphic_file(outfile, results):
    plot = plot_graph(results)
    plt.savefig(outfile, bbox_inches='tight')


# Use matplotlib tools to generate a plot of mean accuracy with shading
# indicating plus or minus one std_dev.

def plot_graph(results):
    pairs = zip(results[1], results[2])
    u_err = [m + s for m, s in pairs]
    l_err = [m - s for m, s in pairs]

    fig, ax = plt.subplots(1)
    ax.plot(results[0], results[1], lw=2, label="mean accuracy",
            color="blue")
    ax.fill_between(results[0], u_err, l_err, facecolor="blue",
            alpha=0.5)
    ax.set_title("accuracy by number of neighbors")
    ax.legend(loc="best")
    ax.set_xlabel('number of nearest neighbors')
    ax.set_ylabel('accuracy')
    ax.grid()
    return fig


if __name__ == '__main__':

    # Parse arguments when run from CLI

    parser = argparse.ArgumentParser()
    parser.add_argument('infile', nargs='?', default="iris")
    parser.add_argument("-o", "--output", nargs='?', const=".knn.out",
        type=argparse.FileType('w'), default=sys.stdout)
    parser.add_argument("-g", "--graph", nargs='?', const=".png",
        default="none")
    parser.add_argument("-n", "--neighbors", type=int, default=100, help=
        "specify a maximum number of neighbors")
    parser.add_argument("-m", "--minneighbors", type=int, default=1, help=
        "specify a minimum number of neighbors")
    parser.add_argument("-k", "--kfold", type=int, default=5, help=
        "specify the number of partitions for cross validation")

    args = parser.parse_args()


    # Acquire data

    if args.infile != "iris":
        # when input file is given, pull data from there
        predictors, responses = get_data(args.infile)
    else:
        # when no input is given, default to using iris data
        iris = datasets.load_iris()
        predictors, responses = iris.data, iris.target


    # Cross validate for various nearest neighbor counts

    err_lists = cross_val_over_neighbors(predictors, responses,
            args.neighbors, args.minneighbors, args.kfold)


    # Output results

    if args.output == ".knn.out":
        put_results(args.infile + args.output, zip(*err_lists))
    else:
        put_results(args.output, zip(*err_lists))

    # Produce graphs

    if args.graph != "none":
        if args.graph == ".png":
            gen_graphic_file(args.infile + args.graph, err_lists)
        else:
            gen_graphic_file(arga.graphic + ".png", err_lists)


