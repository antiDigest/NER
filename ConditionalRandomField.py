
import pandas as pd
import numpy as np
from DataSet import DataSet
from Entities import *


class ConditionalRandomField:

    class Chain:

        def __init__(self, sentence, tags, pi, states):
            assert type(sentence) == list, "Sentence should be a list."
            assert type(tags) == list, "Tags should be a list."

            self.T = len(sentence)
            self.labels = tags
            self.sentence = sentence
            self.pi = pi
            self.states = states

        def __str__(self):
            return " ".join(self.sentence) + ": " + " ".join(self.labels)

        def __repr__(self):
            return " ".join(self.sentence) + ": " + " ".join(self.labels)

        def probability(self, weights):
            """
                    For Numerator:
                    # Sum out the feature map and take an exponential

                    For denominator:
                    # You need to sum over all possible state sequences,
                    # or all possible label sequences with the given sentence. Which needs
                    # a loopy belief propagation over the graph
                    #    OR
                    # You can just calculate the FORWARD and sum out the forward from the
                    # last layer !
            """

            num = np.zeros(len(entities.keys()))  # Numerator
            for t in xrange(0, self.T):
                num = num + np.multiply(weights, self.featureMap(t))

            Z = self.forward(weights)  # Denominator

            P = np.exp(num) / Z
            return P / sum(P)

        def featureMap(self, t):
            features = np.zeros(len(entities.keys()))
            features[getEntity(self.labels[t])] = 1
            return features

        def forward(self, weights):
            alpha = np.zeros((self.T, self.states))
            alpha[0, :] = self.pi

            for t in xrange(1, self.T):
                for state in xrange(0, self.states):
                    alpha[t, state] = sum(
                        alpha[t - 1, :] * np.exp(sum(weights * self.featureMap(t - 1))))

            return sum(alpha[-1, :])

    def __init__(self, dataset):
        self.data = dataset
        self.featureSize = len(entities.keys())
        self.weights = np.ones(self.featureSize)
        self.chains = []

    def getChains(self):
        for row in self.data.iterate():
            chain = self.Chain(row[0], row[1],
                               self.data.startProbability(), self.featureSize)
            self.chains.append(chain)

        return self.chains

    def train(self, alpha=0.1):

        self.getChains()

        featureCount = np.zeros(self.featureSize)
        for chain in self.chains:
            features = np.zeros(self.featureSize)
            for t in xrange(0, chain.T):
                features = features + chain.featureMap(t)

            featureCount += features
            # break

        empirical = featureCount
        print empirical

        for its in xrange(0, 1000):

            chainProb = 0
            for chain in self.chains:
                p = chain.probability(self.weights)
                print p
                features = np.zeros(self.featureSize)
                for t in xrange(0, chain.T):
                    features = features + chain.featureMap(t)

                featureCount += features
                chainProb = chainProb + p * featureCount
                print chainProb

            self.weights = self.weights + \
                (alpha * (empirical - chainProb)) - \
                self.regularize(self.weights)
            print self.weights

            alpha = 2 / (2 + its)

    def regularize(self, weights):
        return sum(np.square(weights)) / (2 * self.featureSize)

if __name__ == '__main__':
    d = DataSet('demo/sample.csv')
    crf = ConditionalRandomField(d)
    crf.train()
    # chains = crf.getChains()
    # for chain in chains:
    #     print chain.sentence
    #     print chain.labels
    #     print chain.probability(crf.weights)
    #     break
