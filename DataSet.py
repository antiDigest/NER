
# @author: Antriksh Agarwal
# Version 0: 4/12/2018

import numpy as np
import pandas as pd
import math
import time
from Entities import *


class DataSet(object):

    def __init__(self, FILE="data/kaggle/ner_dataset.csv"):
        self.source = self.data = pd.read_csv(
            FILE, header=0, dtype={'Sentence #': str})
        self.sentences = []
        self.labels = []
        self.club()
        self.preprocess()

    def club(self):
        self.data.fillna(method='ffill', inplace=True)

    def preprocess(self):
        self.data = self.data.groupby("Sentence #").agg({'Word': lambda x: ";".join(x),
                                                         'Tag': lambda x: ";".join(x)})

    def add(self, sentence, label):
        self.sentences.append(sentence)
        self.labels.append(label)

    def iterate(self):
        for row in self.data.to_records():
            yield row[2].split(';'), row[1].split(';')

    def startProbability(self):
        pi = np.zeros(len(entities.keys()))
        for row in self.iterate():
            pi[getEntity(row[1][0])] += 1

        return pi / sum(pi)

    def to_records(self):
        return self.data.to_records()

if __name__ == '__main__':
    start = time.time()
    d = DataSet()
    print time.time() - start
    for row in d.iterate():
        print row
        break
