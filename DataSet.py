
# @author: Antriksh Agarwal
# Version 0: 4/12/2018

from __future__ import print_function
import numpy as np
import pandas as pd
import math
import time
from collections import OrderedDict, Counter
from utils import *


class DataSet(object):
    """
        DataSet class:
        * Imports dataset from file
        * converts the dataset into <observation, tags> tuples
        * provides an iteration over the tuples
        * to_records returns the list of tuples
    """

    def __init__(self, FILE="data/kaggle/ner_dataset.csv", verbose=False):
        self.source = pd.read_csv(
            "data/kaggle/ner_dataset.csv", header=0, dtype={'Sentence #': str})
        self.data = pd.read_csv(
            FILE, header=0, dtype={'Sentence #': str})
        self.verbose = verbose
        self.sentences = []
        self.labels = []
        self.club()
        self.preprocess()
        self.M = len(self.source.index)
        self.unigrams()
        self.unipos()
        self.transition()

    def club(self):
        if self.verbose:
            logger(
                "[INFO]: Handling missing values: filling inplace copying from top, most recent")
        self.data.fillna(method='ffill', inplace=True)

    def preprocess(self):
        if self.verbose:
            logger("[INFO]: Grouping by sentence number")
        self.data = self.data.groupby("Sentence #").agg({'Word': lambda x: ";".join(x),
                                                         'Tag': lambda x: ";".join(x),
                                                         'POS': lambda x: ";".join(x)})[['Word', 'Tag', 'POS']]

        # logger(self.data)

    def add(self, sentence, label):
        self.sentences.append(sentence)
        self.labels.append(label)

    def iterate(self):
        for row in self.data.to_records():
            yield row[1].split(';'), row[2].split(';'), row[3].split(';')

    def startProbability(self):
        if self.verbose:
            logger(
                "[INFO]: Calculating initial starting probabilities from the dataset")

        self.pi = np.zeros(len(entities.keys()))
        entityList = sorted(entities.keys())
        for tagIndex, tag in enumerate(entityList):
            num = self.source[self.source['Tag'].str.contains(tag)]
            self.pi[tagIndex] = len(num.index)

        self.pi = (self.pi / sum(self.pi))
        return self.pi

    def to_records(self):
        return self.data.to_records()

    def rows(self):
        return len(self.data.index)

    def unigrams(self):
        if self.verbose:
            logger("[INFO]: Extracting unigrams...")
        self.unigrams = list(self.source['Word'].unique())

    def unipos(self):
        if self.verbose:
            logger("[INFO]: Extracting uni POS tags...")
        self.unipos = list(self.source['POS'].unique())

    def transition(self):
        if self.verbose:
            logger("[INFO]: Pre-Calculating label transition probabilities...")

        entityList = sorted(entities.keys())
        S = len(entityList)
        transProb = np.zeros((S, S))
        self.tagCount = np.zeros(S)

        for tagIndex, tag in enumerate(entityList):
            num = self.source[self.source['Tag'].str.contains(tag)]
            self.tagCount[tagIndex] = len(num.index)

            indexes = num.index.values
            indexes = [index + 1 for index in indexes if index < self.M - 1]

            for nextTag in entityList:
                nextNum = self.source.iloc[indexes, :][
                    self.source.iloc[indexes, :]['Tag'].str.contains(nextTag)]
                transProb[tagIndex, entityList.index(
                    nextTag)] = len(nextNum.index)

            if len(num.index) != 0:
                transProb[tagIndex, :] = transProb[
                    tagIndex, :] / len(num.index)
            else:
                transProb[tagIndex, :] = 0

        self.transProb = transProb

        # self.emission = {}
        # for word in self.unigrams:
        #     self.emission[word] = self.source[self.source['Word'] == word]

    def emission(self, label, word):
        count_tag = self.tagCount[label]

        label = entities.keys()[entities.values().index(label)]

        entityList = sorted(entities.keys())

        emission = self.source[self.source['Word'] == word]
        emissionCount = float(len(
            emission[emission['Tag'].str.contains(label)].index))

        if count_tag == 0:
            return 0.
        return emissionCount / count_tag


if __name__ == '__main__':
    print("INIT")
    start = time.time()
    d = DataSet('demo/sample.csv')
    print(time.time() - start)

    print("CALCULATING PROBABILITIES")
    # logger(d.unigrams)
    # start = time.time()
    # transProb = d.transition()
    # logger(time.time() - start)

    d.startProbability()
    print(d.transProb)

    print(d.pi)

    start = time.time()
    print(d.emission(2, 'Superdome'))
    print(time.time() - start)

    # logger(emission)
