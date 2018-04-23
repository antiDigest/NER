
from __future__ import print_function
import pandas as pd
import numpy as np
from DataSet import DataSet
from utils import *
from features import *
import time
import argparse


class ConditionalRandomField(object):

    class Chain(object):

        def __init__(self, sentence, tags, pos, pi, states, dataset, verbose=False):
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
            self.verbose = verbose
            self.features = []

            if self.verbose:
                logger(self.__str__())

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

        def featureMap(self, word, label):
            # wordindex = self.sentence.index(word)
            # if label == self.labels[wordindex] and self.features == []:
            #     features = getFeatureMap(self.sentence, self.pos,
            #                              self.labels, label, word, self.dataset)
            #     self.features = features
            #     return self.features
            # elif label == self.labels[wordindex] and self.features != []:
            #     logger(self.features)
            #     return self.features

            features = getFeatureMap(self.sentence, self.pos,
                                     self.labels, label, word, self.dataset)

            # if self.verbose:
            #     logger("FEATURES: " + features)
            return features

        def forward(self, weights):
            alpha = np.zeros((self.T, self.states))
            alpha[0, :] = self.pi

            for t in xrange(1, self.T):
                # logger("[VECTOR]: ALPHA[t-1] = " + str(alpha[t - 1, :]))
                for state in xrange(0, self.states):
                    f = weights * \
                        self.featureMap(
                            self.sentence[t - 1], entities.keys()[state])
                    # logger("[FEATURE]: " + str(f))
                    # In log-sum-exp space instead of sum-exp state
                    alpha[t, state] = sum(alpha[t - 1, :] * sum(f))
                    # logger("[VECTOR]: ALPHA = " + str(alpha[t, :]))

                logger("[VECTOR]: ALPHA = " + str(alpha[t, :]))

                # Normalised Probability
                # if t != self.T:
                #     alpha[t, :] = alpha[t, :] / sum(alpha[t, :])

            return np.log(sum(alpha[-1, :]))

    def __init__(self, dataset, verbose=False):
        self.data = dataset
        self.M = dataset.rows()
        self.featureSize = NUMFEATURES
        self.weights = np.ones(self.featureSize)
        self.chains = []
        self.verbose = verbose

    def getChains(self):
        if self.verbose:
            logger("[INFO]: Initialising all the CRF Chains...")

        startProb = self.data.startProbability()
        for row in self.data.iterate():
            chain = self.Chain(row[0], row[1], row[2],
                               startProb, self.featureSize, self.data)
            self.chains.append(chain)

        return self.chains

    def train(self, alpha=0.1):

        start = "[INFO]:[TRAINING]:"

        if self.verbose:
            logger(start + " Training CRF Model...")
            logger(start + " Alpha=" + str(alpha))

        self.getChains()

        if self.verbose:
            logger(start + " Number of training instances=" +
                   str(len(self.chains)))

        global featureCount
        featureCount = np.zeros(self.featureSize)

        def extract(chain):

            # global featureCount
            features = np.zeros(self.featureSize)
            for t in xrange(0, chain.T):
                features = features + \
                    chain.featureMap(
                        chain.sentence[t - 1], chain.labels[t - 1])
            # featureCount += features

            # if self.verbose:
            #     chainindex = self.chains.index(chain)
            #     if chainindex % 10 == 0:
            #         logger(start + str(chainindex) + "/" +
            #                str(len(self.chains)) + "[VECTOR]: Empirical Probability")
            # logger(start + str(featureCount))
            return features

        def reduceF(features):
            return np.sum(features)

        # Trying to make it run faster

        # from multiprocessing import Process
        # import os
        # processes = []
        # for i in xrange(0, 4):
        #     chains = self.chains[(i * 1):((len(self.chains) / 4) * (i + 1))]
        #     p = Process(target=extract, args=(chains,))
        #     p.start()
        #     processes.append(p)

        # for i in xrange(0, 4):
        #     processes[i].join()

        from pathos.multiprocessing import ProcessingPool as Pool

        pool = Pool(3)
        data = pool.map(extract, self.chains)
        featureCount = pool.map(reduceF, data)
        pool.close()
        pool.join()

        # break

        empirical = featureCount
        if self.verbose:
            logger(start + "[VECTOR]: Empirical Probability")
            logger(start + str(empirical))

        # self.weights = empirical

        its = 0
        chainProb = 0
        # while sum(empirical - chainProb) > 0.00001:
        for its in xrange(0, 5):

            chainProb = 0
            for chain in self.chains:
                p = chain.forward(self.weights)
                if self.verbose:
                    logger(start + " PROBABILITY: " + str(p))
                chainProb = chainProb + p

            self.weights = np.log(np.exp(self.weights) +
                                  (alpha * (sum(empirical) - chainProb) / np.exp(self.weights)))  # - \
            # np.log(self.regularize(self.weights))

            if self.verbose:
                logger(start + "[VECTOR]: WEIGHTS")
                logger(start + str(self.weights))

            alpha = 2 / (2 + its)
            # its += 1
            if its == 4:
                break

    def regularize(self, weights):
        # logger(sum(np.square(weights)) / (2 * self.featureSize))
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

    #         # logger viterbi[t]

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

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true")
    args = parser.parse_args()
    if args.verbose:
        logger("[INFO] Giving verbose output")

    start = time.time()
    d = DataSet(verbose=args.verbose)
    print("[INFO]: Time Taken = " + str(time.time() - start))
    start = time.time()
    crf = ConditionalRandomField(d, verbose=args.verbose)
    print("[INFO]: Time Taken = " + str(time.time() - start))
    # crf.train()
    # chains = crf.getChains()
    start = time.time()
    crf.train()
    print("[INFO]: Time Taken = " + str(time.time() - start))

    # for chain in chains:
    #     print(" ".join(chain.sentence))
    #     print(chain.forward(crf.weights))
    #     # print crf.viterbi(chain)
    #     break
