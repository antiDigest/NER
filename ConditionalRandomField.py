
from __future__ import print_function
import pandas as pd
import numpy as np
from DataSet import DataSet
from utils import *


class ConditionalRandomField(object):

    class Chain(object):

        def __init__(self, sentence, tags, pos, pi, states, dataset):
            assert type(sentence) == list, "Sentence should be a list."
            assert type(tags) == list, "Tags should be a list."
            assert type(pos) == list, "POS should be a list."

            self.T = len(sentence)
            self.labels = tags
            self.pos = pos
            self.sentence = sentence
            self.pi = pi
            self.states = len(entities.keys())
            self.dataset = dataset

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

            num = np.zeros(len(weights))  # Numerator
            for t in xrange(0, self.T):
                num = num + np.multiply(weights, self.featureMap(t))

            Z = self.forward(weights)  # Denominator

            return np.exp(num) / Z
            # return P / sum(P)

        def featureMap(self, word):
            return getFeatureMap(self.sentence, self.pos, self.labels, word, self.dataset)

        def forward(self, weights):
            alpha = np.zeros((self.T, self.states))
            alpha[0, :] = self.pi

            for t in xrange(1, self.T):
                f = weights * self.featureMap(self.sentence[t - 1])
                for state in xrange(0, self.states):
                    alpha[t, state] = sum(alpha[t - 1, :] * sum(f))

            print("FORWARD: " + str(sum(alpha[-1, :])))
            return sum(alpha[-1, :])

    def __init__(self, dataset):
        self.data = dataset
        self.M = dataset.rows()
        self.featureSize = NUMFEATURES
        self.weights = np.ones(self.featureSize)
        self.chains = []

    def getChains(self):
        for row in self.data.iterate():
            chain = self.Chain(row[0], row[1], row[2], self.data.startProbability(
            ), self.featureSize, self.data)
            self.chains.append(chain)

        return self.chains

    def train(self, alpha=0.1):

        self.getChains()

        featureCount = np.zeros(self.featureSize)
        for chain in self.chains:
            features = np.zeros(self.featureSize)
            for t in xrange(0, chain.T):
                features = features + chain.featureMap(chain.sentence[t])
            featureCount += features
            # break

        empirical = featureCount
        print(empirical)

        # self.weights = empirical

        its = 0
        chainProb = 0
        while sum(empirical - chainProb) > 0.00001:
            # for its in xrange(0, 1000):

            chainProb = 0
            for chain in self.chains:
                p = chain.forward(self.weights)

                # features = np.zeros(self.featureSize)
                # for t in xrange(0, chain.T):
                #     features = features + chain.featureMap(chain.sentence[t])

                # featureCount += features
                chainProb = chainProb + p
                # print (chainProb)

            self.weights = self.weights + \
                (alpha * (sum(empirical) - chainProb)) - \
                self.regularize(self.weights)
            print(self.weights)

            alpha = 2 / (2 + its)
            # its += 1
            if its == 4:
                break

    def regularize(self, weights):
        print(sum(np.square(weights)) / (2 * self.featureSize))
        return sum(np.square(weights)) / (2 * self.featureSize)

    # def viterbi(self, chain):
    #     """
    #         The State sequence generator, takes as input the probability of next state given the first
    #         and the probability of having one state given an observation and maximises the sequence that
    #         can be made.
    #         This function generates the viterbi sequence for a set of observations.
    #     """

    #     viterbi = [{}]

    #     for state in xrange(0, self.states):
    #         prob = np.exp(sum(weights * self.featureMap(t - 1))
    #                       ) * self.pi[state]
    #         viterbi[0][state] = {'prob': prob, 'prev': None}

    #     for t in range(1, self.T):
    #         viterbi.append({})
    #         for state in self.states:
    #             # max(Prev_prob * trans_prob, prev)
    #             maxProb, prevState = max([(np.exp(sum(weights * self.featureMap(t - 1))) *
    #                                        viterbi[t - 1][prev]['prob'], prev)
    #                                       for prev in states], key=lambda i: i[0])
    #             # max(prev_prob * trans_prob, prev_prob) * emission_prob
    #             # emission_prob: probability of observation given the label
    #             maxProb = maxProb * \
    #                 np.exp(sum(weights * self.featureMap(t - 1)))

    #             viterbi[t][state] = {'prob': maxProb, 'prev': prevState}

    #         # print viterbi[t]

    #     max_elem, max_prob, max_prev = max([(key, value["prob"], value['prev'])
    # for key, value in viterbi[-1].items()], key=lambda i: i[1])

    #     sequence = []
    #     sequence.insert(0, max_elem)

    #     k = len(viterbi) - 2
    #     while max_prev != None:
    #         sequence.insert(0, max_prev)
    #         max_prev = viterbi[k][max_prev]['prev']
    #         k -= 1

    #     return sequence

if __name__ == '__main__':
    d = DataSet('demo/sample.csv')
    crf = ConditionalRandomField(d)
    # crf.train()
    # chains = crf.getChains()
    crf.train()
    # for chain in chains:
    #     print(" ".join(chain.sentence))
    #     print(chain.forward(crf.weights))
    #     # print crf.viterbi(chain)
    #     break
