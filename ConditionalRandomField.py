
from __future__ import print_function
import pandas as pd
import numpy as np
from DataSet import DataSet
from utils import *
from features import *
import time
import argparse
import itertools
from scipy.optimize import minimize, fmin_l_bfgs_b
import scipy
from scipy import misc, optimize
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def log_dot_vm(loga, logM):
    return misc.logsumexp(loga.reshape(loga.shape + (1,)) + logM, axis=0)


def log_dot_mv(logM, logb):
    return misc.logsumexp(logM + logb.reshape((1,) + logb.shape), axis=1)


class ConditionalRandomField(object):

    class Chain(object):

        def __init__(self, sentence, tags, pos, pi, dataset, featureSize, verbose=False):
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
            self.featureSize = featureSize

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

        def all_features(self):

            F = np.zeros((self.T, self.states,
                          self.states, self.featureSize))

            for t in xrange(1, self.T):
                for prev_state in xrange(0, self.states):
                    for state in xrange(0, self.states):
                        F[t, prev_state, state] = self.featureMap(self.sentence[t - 1],
                                                                  entities.keys()[state], entities.keys()[prev_state])

            logger("[VECTOR]: FEATURE SHAPE = " + str(F.shape), print_it=False)

            return F

        def forward(self, M):
            alpha = np.NINF * np.ones((self.T, self.states))
            alpha[0, :] = self.pi

            for t in xrange(1, self.T):
                alpha[t] = log_dot_vm(alpha[t - 1], M[t - 1])

            logger("[VECTOR]: ALPHA = " + str(alpha[-1]), print_it=False)

            return (alpha, alpha[-1])

        def viterbi(self, weights):

            viterbi = [{}]

            for state in xrange(0, self.states):
                prob = np.exp(sum(weights * self.featureMap(self.sentence[0],
                                                            entities.keys()[state], -1))
                              ) * self.pi[state]
                viterbi[0][state] = {'prob': prob, 'prev': None}

            # logger viterbi[t]

            for t in range(1, self.T):
                viterbi.append({})
                for state in xrange(0, self.states):
                    # max(Prev_prob * trans_prob, prev)
                    maxProb, prevState = max([(np.exp(sum(weights * self.featureMap(self.sentence[t],
                                                                                    entities.keys()[
                        state],
                        entities.keys()[prev]))) * viterbi[t - 1][prev]['prob'], prev)
                        for prev in xrange(0, self.states)], key=lambda i: i[0])

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

    def __init__(self, dataset, verbose=False, sigma=10):
        self.data = dataset
        self.M = dataset.rows()
        self.featureSize = NUMFEATURES
        self.weights = np.random.rand(NUMFEATURES)
        self.chains = []
        self.verbose = verbose
        self.v = sigma ** 2
        self.v2 = self.v * 2

    def getChains(self):
        if self.verbose:
            logger("[INFO]: Initialising all the CRF Chains...")

        self.pi = self.data.startProbability()
        for row in self.data.iterate():
            chain = self.Chain(row[0], row[1], row[2],
                               self.pi, self.data, self.featureSize)
            self.chains.append(chain)

        # print(self.chains)

        return self.chains

    def train(self, alpha=0.1):

        start = "[INFO]:[TRAINING]:"
        logger(start + " Training CRF Model...", print_it=self.verbose)
        logger(start + " Alpha=" + str(alpha), print_it=self.verbose)

        self.getChains()

        logger(start + " Number of training instances=" +
               str(len(self.chains)), print_it=self.verbose)

        global featureCount
        featureCount = np.zeros(self.featureSize)

        def trainer(weights, chains):
            logger(start + "[VECTOR]: WEIGHTS: " + str(weights))

            likelihood = 0
            derivative = np.zeros(len(weights))
            logger("[INFO]: INIT CHAINS")
            for chain in chains:
                all_features = chain.all_features()
                log_M = np.dot(all_features, weights)
                log_alphas, last = chain.forward(log_M)

                time, state = log_alphas.shape

                log_alphas1 = log_alphas.reshape(time, state, 1)
                log_Z = misc.logsumexp(last)
                log_probs = log_alphas1 + log_M - log_Z
                log_probs = log_probs.reshape(log_probs.shape + (1,))

                yp_vec_ids = [getEntity(label) for label in chain.labels[:-1]]
                y_vec_ids = [getEntity(label) for label in chain.labels[1:]]

                emp = np.array([all_features[range(chain.T), row, index]
                                for row, index in zip(yp_vec_ids, y_vec_ids)])
                m = np.array([log_M[range(chain.T), row, index]
                              for row, index in zip(yp_vec_ids, y_vec_ids)])

                exp_features = np.sum(
                    np.exp(log_probs) * all_features, axis=(0, 1, 2))
                emp_features = np.sum(emp, axis=(0, 1))

                likelihood += np.sum(m, axis=(0, 1)) - log_Z
                derivative += emp_features - exp_features

            l = -(likelihood - self.regulariser(weights))
            J = -(derivative - self.regulariser_deriv(weights))

            logger(start + " COST: " + str(l), print_it=self.verbose)
            logger(start + " GRADIENT: " + str(J), print_it=self.verbose)

            return l, J

        # value = fmin_l_bfgs_b(trainer, self.weights)
        # its = 0
        for its in xrange(0, 100):
            for chain in self.iterate():
                res, _, _ = fmin_l_bfgs_b(
                    trainer, self.weights, args=(chain, ), disp=True, maxiter=5)
                self.weights = res
                logger(start + "[VECTOR]: WEIGHTS: " + str(self.weights))

    def regulariser(self, w):
        return np.sum(w ** 2) / self.v2

    def regulariser_deriv(self, w):
        return np.sum(w) / self.v

    def viterbi(self, chain):
        """
            The State sequence generator, takes as input the probability of next state given the first
            and the probability of having one state given an observation and maximises the sequence that
            can be made.
            This function generates the viterbi sequence for a set of observations.
        """

        return chain.viterbi(self.weights)

    def iterate(self, batch_size=4):
        ch = len(self.chains)
        for val in xrange(0, ch, batch_size):
            yield self.chains[val:val + batch_size]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", help="For verbose output",
                        action="store_true")
    parser.add_argument("-d", "--demo", help="For learning on the sample dataset",
                        action="store_true")
    args = parser.parse_args()
    if args.verbose:
        logger("[INFO] Giving verbose output")

    start = time.time()
    if args.demo:
        d = DataSet(FILE='demo/sample_test.csv', verbose=args.verbose)
    else:
        d = DataSet(verbose=args.verbose)
    print("[INFO]: Time Taken = " + str(time.time() - start))
    start = time.time()
    crf = ConditionalRandomField(d, verbose=args.verbose)
    print("[INFO]: Time Taken = " + str(time.time() - start))
    # crf.train()
    # chains = crf.getChains()
    # start = time.time()
    # crf.train()
    # print("[INFO]: Time Taken = " + str(time.time() - start))

    # crf.weights = np.array(
    #     [0.8003172,   0.48495405,  3.79160371, -0.02069089,  0.67821951,  0.29449089,
    #      0.00691485,  0.37848588,  0.55161599])

    # crf.weights = np.array(
    #     [0.51581581, 0.1683381,  0.70498652, 0.09672789, 0.08480342, 0.89593315,
    #      0.21023996, 0.95635682, 0.7429099])

    crf.weights = np.array(
        [8.30669958e-01, 5.15306809e-01, 2.80062932e+00, 9.66186060e-03,
         6.78219508e-01, 2.90595573e-01, 1.17759415e-04, 3.47312865e-01,
         5.81968749e-01])

    crf.getChains()
    print(crf.pi)
    output_y = []
    target_y = []
    for chain in crf.chains:
        logger(" ".join(chain.sentence))
        logger(" ".join(chain.labels))
        sequence = crf.viterbi(chain)
        target_y += list(chain.labels)
        sequence = [entities.keys()[entities.values().index(s)]
                    for s in sequence]
        output_y += list(sequence)
        logger(" ".join(sequence))

    logger(confusion_matrix(target_y, output_y))
    logger(classification_report(
        target_y, output_y, target_names=entities.keys()))
