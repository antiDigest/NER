
from __future__ import print_function
import pandas as pd
import numpy as np
from DataSet import DataSet
from utils import *
from features import *
import time
import argparse
import itertools
from scipy.optimize import minimize


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

        def featureMap(self, word, label, prev_label):

            features = getFeatureMap(self.sentence, self.pos,
                                     self.labels, label, prev_label, word, self.dataset)

            # if self.verbose:
            #     logger("FEATURES: " + features)
            return features

        def forward(self, weights):
            alpha = np.zeros((self.T, self.states))
            alpha[0, :] = self.pi

            for t in xrange(1, self.T):
                # logger("[VECTOR]: ALPHA[t-1] = " + str(alpha[t - 1, :]))
                # alpha[t - 1, :] = alpha[t - 1, :] / \
                #     np.sum(alpha[t - 1, :])  # NORMALIZE
                for state in xrange(0, self.states):
                    # next_alpha = np.zeros(self.states)
                    # for prev_state in xrange(0, self.states):
                    F = [self.featureMap(self.sentence[t - 1],
                                         entities.keys()[state],
                                         entities.keys()[prev_state])
                         for prev_state in xrange(0, self.states)]

                    # print("[FEATURE]: " + str(F))
                    f = np.array([np.sum(weights * feature) for feature in F])
                    # logger("[FEATURE NORM]: " + str(f))
                    maxf = np.max(f)
                    # logger("[MAXIMUM]: " + str(maxf))
                    if maxf > 0:
                        f = f / maxf
                    f = np.exp(f)
                    # logger("[FEATURE NORM]: " + str(f))
                    # In log-sum-exp space instead of sum-exp state
                    # logger("[ALPHA]: " + str(alpha[t - 1, :]))
                    next_alpha = alpha[t - 1, :] * np.log(f)
                    max_alpha = np.max(next_alpha)
                    if max_alpha > 0:
                        # f = f / maxf
                        next_alpha = next_alpha / max_alpha
                    alpha[t, state] = np.sum(next_alpha)

            # logger("[VECTOR]: ALPHA = " + str(alpha[self.T - 1, :]))

            return sum(alpha[-1, :])

        def viterbi(self, weights):

            viterbi = [{}]

            for state in xrange(0, self.states):
                prob = np.exp(sum(weights * self.featureMap(t - 1))
                              ) * self.pi[state]
                viterbi[0][state] = {'prob': prob, 'prev': None}

            # logger viterbi[t]

            for t in range(1, self.T):
                viterbi.append({})
                for state in xrange(0, self.states):
                    # max(Prev_prob * trans_prob, prev)
                    f = sum(
                        weights * chain.featureMap(chain.sentence[t - 1], chain.labels[t - 1]))
                    maxProb, prevState = max([(np.exp(f) * viterbi[t - 1][prev]['prob'], prev)
                                              for prev in states], key=lambda i: i[0])

                    viterbi[t][state] = {'prob': maxProb, 'prev': prevState}

            max_elem, max_prob, max_prev = max([(key, value["prob"], value[
                'prev']) for key, value in viterbi[-1].items()],
                key=lambda i: i[1])

            sequence = []
            sequence.insert(0, max_elem)

            k = len(viterbi) - 2
            while max_prev != None:
                sequence.insert(0, max_prev)
                max_prev = viterbi[k][max_prev]['prev']
                k -= 1

            return sequence

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

        # print(self.chains)

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
            features = np.zeros(self.featureSize)
            for t in xrange(0, chain.T):
                if t == 0:
                    features = chain.featureMap(
                        chain.sentence[t], chain.labels[t], -1)
                else:
                    features = features + \
                        chain.featureMap(
                            chain.sentence[t], chain.labels[t], chain.labels[t - 1])

            # logger("[FEATURES]: " + str(features), print_it=False)
            return features

        def partition(mapped_values):
            for index, value in enumerate(mapped_values):
                if index == 0:
                    sumValue = np.array(value)
                else:
                    sumValue = np.add(sumValue, np.array(value))
            return sumValue

        # Trying to make it run faster

        from pathos.multiprocessing import ProcessingPool as Pool

        pool = Pool(10)
        data = pool.map(extract, self.chains)
        featureCount = partition(data)

        empirical = featureCount
        # empirical = 1.
        if self.verbose:
            logger(start + "[VECTOR]: Empirical Probability")
            logger(start + str(empirical))

        its = 0
        chainProb = 0
        pool1 = Pool(10)
        # while np.sum(empirical) - chainProb > 0.00001:

        def trainer(weights):

            chainProb = 0

            def chainExtract(chain):
                # p = 0
                # for chain in self.chains:
                p = chain.forward(weights)
                # if self.verbose:
                #     logger(start + " PROBABILITY: " + str(p))
                return p

            data = pool1.map(chainExtract, self.chains)
            chainProb = partition(data)
            # chainProb = p
            if self.verbose:
                logger(start + " CHAIN PROBABILITY: " + str(chainProb))
                logger(start + " EMPERICAL PROBABILITY: " +
                       str(np.sum(empirical)))
                logger(start + "[VECTOR]: WEIGHTS: " + str(weights))

            J = np.array(empirical) - chainProb
            # % Gradient
            # J = J / np.exp(weights)

            # cost = alpha * J

            return np.sum(empirical) - chainProb, J

        # value = fmin_l_bfgs_b(trainer, self.weights)
        res = minimize(trainer, self.weights,
                       method='L-BFGS-B', jac=True,
                       options={'ftol': 1e-4, 'disp': True, 'maxiter': 1000})

        print(res.x)
        print(res.success)

        # self.weights = np.log(np.exp(self.weights) +
        #                       (alpha * J)) - self.regularize(self.weights)

        # if self.verbose:
        #     logger(start + "[VECTOR]: UPDATED WEIGHTS")
        #     logger(start + str(self.weights))

        # alpha = 2 / (2 + its)
        # its += 1

        pool1.close()
        pool1.join()

    def regularize(self, weights):
        logger("L2 NORM: " + str(sum(np.square(weights)) * 0.01))
        return (0.3 * sum(np.square(weights)))

    def viterbi(self, chain):
        """
            The State sequence generator, takes as input the probability of next state given the first
            and the probability of having one state given an observation and maximises the sequence that
            can be made.
            This function generates the viterbi sequence for a set of observations.
        """

        return chain.viterbi(self.weights)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true")
    parser.add_argument("-d", "--demo", help="increase output verbosity",
                        action="store_true")
    args = parser.parse_args()
    if args.verbose:
        logger("[INFO] Giving verbose output")

    start = time.time()
    if args.demo:
        d = DataSet(FILE='demo/sample.csv', verbose=args.verbose)
    else:
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

    for chain in crf.chains:
        print(" ".join(chain.sentence))
        print(crf.viterbi(chain))
        break
