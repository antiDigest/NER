
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

    def __init__(self, FILE="data/kaggle/ner_dataset.csv"):
        self.source = self.data = pd.read_csv(
            FILE, header=0, dtype={'Sentence #': str})
        self.sentences = []
        self.labels = []
        self.club()
        self.preprocess()
        self.M = len(self.source.index)

    def club(self):
        self.data.fillna(method='ffill', inplace=True)

    def preprocess(self):
        self.data = self.data.groupby("Sentence #").agg({'Word': lambda x: ";".join(x),
                                                         'Tag': lambda x: ";".join(x),
                                                         'POS': lambda x: ";".join(x)})[['Word', 'Tag', 'POS']]

    def add(self, sentence, label):
        self.sentences.append(sentence)
        self.labels.append(label)

    def iterate(self):
        for row in self.data.to_records():
            yield row[1].split(';'), row[2].split(';'), row[3].split(';')

    def startProbability(self):
        pi = np.zeros(len(entities.keys()))
        for row in self.iterate():
            pi[getEntity(row[1][0])] += 1
        return (pi / sum(pi))

    def to_records(self):
        return self.data.to_records()

    def rows(self):
        return len(self.data.index)

    def unigrams(self):
        wordSet = []
        for row in self.iterate():
            wordSet += row[0]

        return list(np.unique(wordSet))

    def unipos(self):
        wordSet = []
        for row in self.iterate():
            wordSet += row[2]

        return list(np.unique(wordSet))

    def transition(self):

        entityList = sorted(entities.keys())
        S = len(entityList)
        transProb = np.zeros((S, S))

        for tagIndex, tag in enumerate(entityList):
            num = self.source[self.source['Tag'].str.contains(tag)]

            indexes = num.index.values
            indexes = [index + 1 for index in indexes if index < self.M - 1]

            for nextTag in entityList:
                nextNum = self.source.iloc[indexes, :][
                    self.source.iloc[indexes, :]['Tag'].str.contains(nextTag)]
                transProb[tagIndex, entityList.index(
                    nextTag)] = len(nextNum.index)

            transProb[tagIndex, :] = transProb[tagIndex, :] / len(num.index)

        self.transProb = transProb
        return transProb

    def emission(self):
        entityList = sorted(entities.keys())
        emission = {}

        emission = self.source[['Word', 'Tag', 'Sentence #']].groupby(
            ['Word', 'Tag']).agg(['count'])

        tagCount = self.source[['Word', 'Tag']].groupby(
            ['Tag']).agg(['count']).rename(columns={'count': 'count'})

        print(tagCount.columns.values)
        print(tagCount)
        print(tagCount['Word']['Tag'])
        print(tagCount['Word']['count'])

        for tagIndex, tag in enumerate(entityList):
            countSum = tagCount[tagCount['Tag'].str.contains(tag)][
                'count'].values
            print(countSum)
            break

        self.emission = emission
        return emission


if __name__ == '__main__':
    print("INIT")
    start = time.time()
    d = DataSet()
    print(time.time() - start)

    print("CALCULATING PROBABILITIES")
    # start = time.time()
    # transProb = d.transition()
    # print(time.time() - start)

    # print(transProb)

    start = time.time()
    emission = d.emission()
    print(time.time() - start)

    # print(emission)
