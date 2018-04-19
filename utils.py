
# @author: Antriksh Agarwal
# Version 0: 4/13/2018

from __future__ import print_function
import numpy as np

# Named Entities:
#   geo = Geographical Entity
#   org = Organization
#   per = Person
#   gpe = Geopolitical Entity
#   tim = Time indicator
#   art = Artifact
#   eve = Event
#   nat = Natural Phenomenon

entities = dict({'O': 0, 'geo': 1, 'org': 2, 'per': 3,
                 'gpe': 4, 'tim': 5, 'art': 6, 'eve': 7, 'nat': 8})

NUMFEATURES = 10


def getEntity(label):
    for entity in entities.keys():
        if entity in label:
            return entities[entity]

    return 0


def getFeatureMap(sentence, pos, word, unigrams, unipos, prevProb, obsProb):
    wordindex = sentence.index(word)
    features = extractFeatures(
        sentence, pos, wordindex, unigrams, unipos, prevProb, obsProb)

    featureMap = np.zeros(NUMFEATURES)
    for index, feature in enumerate(features.keys()):
        featureMap[index] = features[feature]

    return featureMap


def extractFeatures(sentence, pos, wordindex, unigrams, unipos, prevProb, obsProb):
    # TODO: how to handle non-binary features
    word = sentence[wordindex]
    word_pos = pos[wordindex]
    try:
        prev_word = unigrams.index(sentence[wordindex - 1])
        prev_pos = unipos.index(pos[wordindex - 1])
        if (wordindex == 0):
            prev_word = -1
            prev_pos = -1
    except ValueError:
        prev_word = -1
        prev_pos = -1

    try:
        next_word = unigrams.index(sentence[wordindex + 1])
        next_pos = unipos.index(pos[wordindex + 1])
    except ValueError and IndexError:
        next_word = -2
        next_pos = -2

    # d = Data

    features = {
        'isupper': word.isupper(),
        'islower': word.islower(),
        'istitle': word.istitle(),
        'isdigit': word.isdigit(),
        'word': unigrams.index(word),
        'next_word': next_word,
        'prev_word': prev_word,
        'pos': unipos.index(word_pos),
        'pos_next': next_pos,
        'pos_prev': prev_pos,
        'prev_state_prob': prevProb,
        'obs_prob': obsProb
    }
    return features
