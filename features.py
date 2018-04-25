
# @author: Antriksh Agarwal
# Version 0: 4/22/2018

from __future__ import print_function
import numpy as np
from nltk.corpus import wordnet as wn
from utils import *


nouns = {x.name().split('.', 1)[0] for x in wn.all_synsets('n')}
NUMFEATURES = 2


def getFeatureMap(sentence, pos, labels, label, prev_label, word, dataset):
    wordindex = sentence.index(word)
    features = extractFeatures(
        sentence, pos, labels, label, prev_label, wordindex, dataset)

    # print("[features]: [features]: " + str(features))

    featureMap = np.array(features.values())

    # print(featureMap)

    return featureMap


def extractFeatures(sentence, pos, labels, label, prev_label, wordindex, dataset):
    # TODO: how to handle non-binary features
    word = sentence[wordindex]
    word_pos = pos[wordindex]
    word_label = getEntity(label)

    if prev_label != -1:
        prev_label = getEntity(prev_label)

    try:
        prev_word = findIndex(
            sentence[wordindex - 1], dataset.unigrams)
    #     prev_pos = findIndex(pos[wordindex - 1], dataset.unipos)
    #     # prev_label = getEntity(labels[wordindex - 1])
        if (wordindex == 0):
            prev_word = -1
    #         prev_pos = -1
    #         # prev_label = -1
    except:
        prev_word = -1
    #     prev_pos = -1
    #     # prev_label = -1

    # try:
    #     next_word = dataset.unigrams.index(sentence[wordindex + 1])
    #     next_pos = dataset.unipos.index(pos[wordindex + 1])
    # except:
    #     next_word = -2
    #     next_pos = -2

    # d = Data
    # print("Current pos tags", pos[wordindex])
    # if next_pos != -2:
    #     print("Next POS", pos[next_pos])
    # else:
    #     print("NO POS NExt")

    features = {
        # 'isupper': word.isupper(),
        # 'islower': word.islower(),
        # 'istitle': word.istitle(),
        # 'isdigit': word.isdigit(),
        # 'word': dataset.unigrams.index(word),
        # 'next_word': next_word,
        # 'prev_word': prev_word,
        # 'pos': dataset.unipos.index(word_pos),
        # 'pos_next': next_pos,
        # 'pos_prev': prev_pos,
        # checking in wordnet for nouns
        # 'isNoun1': isNoun(word),
        # if next word is Verb
        # if Next POS is Verb
        # 'is_next_verb': next_pos != -2 or (pos[next_pos] == "VBZ")
        # or (next_pos != -2 or (pos[next_pos] == "VH"
        #                        or pos[next_pos] == "VHD"
        #                        or pos[next_pos] == "VHN"))
        # or (next_pos != -2 or (pos[next_pos] == "VV" or pos[next_pos] == "VVD")),
        # 'isCompany': (word.isupper() or word[0].isupper()) and (sentence[wordindex + 1].lower() == "inc" or sentence[wordindex + 1].lower() == "inc."),
        # 'isOrg': (word.isupper() or word[0].isupper()) and (sentence[wordindex + 1].lower() == "org" or sentence[wordindex + 1].lower() == "org."),
        # 'isCity1': (word.isupper() or word[0].isupper()) and "city" in sentence[wordindex + 1].lower(),
        # 'isCounty1': (word.isupper() or word[0].isupper()) and "county" in sentence[wordindex + 1].lower(),
        # 'isCity2': (word.isupper() or word[0].isupper()) and "city of" in sentence[wordindex - 1].lower(),
        # 'isCounty2': (word.isupper() or word[0].isupper()) and "county of" in sentence[wordindex - 1].lower(),
        'prev_state_prob': 47959. / 1048576.,
        'obs_prob': 0
    }

    if prev_label != -1 and prev_word != '.':
        # print(str(prev_label) + " -> " + str(word_label))
        features['prev_state_prob'] = dataset.transProb[prev_label, word_label]
        # print(features['prev_state_prob'])
    else:
        # print(str(prev_label) + " -> " + str(word_label))
        features['prev_state_prob'] = dataset.pi[word_label]
        # print(features['prev_state_prob'])

    obs_prob = dataset.emission(word_label, word)
    if obs_prob != -1:
        features['obs_prob'] = obs_prob

    # logging.info(str(features))
    return features


def isNoun(word):
    if word in nouns:
        return True
    else:
        return False
