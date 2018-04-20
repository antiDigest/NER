
# @author: Antriksh Agarwal
# Version 0: 4/13/2018

from __future__ import print_function
import numpy as np
from nltk.corpus import wordnet as wn
from PyDictionary import PyDictionary

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
nouns = {x.name().split('.', 1)[0] for x in wn.all_synsets('n')}
dictionary = PyDictionary()
NUMFEATURES = 21


def getEntity(label):

    if label == []:
        return -1
    else:
        label = label[0]

    for entity in entities.keys():
        if entity in label:
            return entity

    return 0


def getFeatureMap(sentence, pos, word, unigrams, unipos, dataset):
    wordindex = sentence.index(word)
    features = extractFeatures(
        sentence, pos, wordindex, unigrams, unipos, dataset)

    featureMap = np.zeros(NUMFEATURES)
    for index, feature in enumerate(features.keys()):
        featureMap[index] = features[feature]

    return featureMap


def extractFeatures(sentence, pos, wordindex, unigrams, unipos, dataset):
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
    # print("Current pos tags", pos[wordindex])
    # if next_pos != -2:
    #     print("Next POS", pos[next_pos])
    # else:
    #     print("NO POS NExt")

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
        # checking in wordnet for nouns
        'isNoun1': isNoun(word),
        # checking in pyDictionary for nouns
        'isNoun2': dictionary.meaning(word) != None and "Noun" in dictionary.meaning(word).keys(),
        # if next word is Verb
        #if Next POS is Verb
        'isNoun3' : next_pos != -2 and (pos[next_pos] == "VBZ"),
        'isNoun4': next_pos != -2 and (pos[next_pos] == "VH" or pos[next_pos] == "VHD" or pos[next_pos] == "VHN"),
        'isNoun5': next_pos != -2 and (pos[next_pos] == "VV" or pos[next_pos] == "VVD"),
        'isCompany': (word.isupper() or word[0].isupper()) and (sentence[wordindex + 1].lower() == "inc" or sentence[wordindex + 1].lower() == "inc."),
        'isOrg': (word.isupper() or word[0].isupper()) and (sentence[wordindex + 1].lower() == "org" or sentence[wordindex + 1].lower() == "org."),
        'isCity1': (word.isupper() or word[0].isupper()) and "city" in sentence[wordindex + 1].lower(),
        'isCounty1': (word.isupper() or word[0].isupper()) and "county" in sentence[wordindex + 1].lower(),
        'isCity2': (word.isupper() or word[0].isupper()) and "city of" in sentence[wordindex - 1].lower(),
        'isCounty2': (word.isupper() or word[0].isupper()) and "county of" in sentence[wordindex - 1].lower(),
        # 'prev_state_prob': prevProb,
        # 'obs_prob': obsProb
    }
    return features


def isNoun(word):
    if word in nouns:
        return True
    else:
        return False
