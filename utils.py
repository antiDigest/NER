
# @author: Antriksh Agarwal
# Version 0: 4/13/2018

import numpy as np

# Named Entities:
# 	geo = Geographical Entity
# 	org = Organization
# 	per = Person
# 	gpe = Geopolitical Entity
# 	tim = Time indicator
# 	art = Artifact
# 	eve = Event
# 	nat = Natural Phenomenon

entities = dict({'O': 0, 'geo': 1, 'org': 2, 'per': 3,
                 'gpe': 4, 'tim': 5, 'art': 6, 'eve': 7, 'nat': 8})

NUMFEATURES = 4


def getEntity(label):
    for entity in entities.keys():
        if entity in label:
            return entities[entity]

    return 0


def getFeatureMap(sentence, wordindex):
    features = extractFeatures(sentence, wordindex)

    featureMap = np.zeros(NUMFEATURES)
    for index, feature in enumerate(features.keys()):
        featureMap[index] = features[feature]

    return featureMap


def extractFeatures(sentence, wordindex):
    # TODO: how to handle non-binary features
    word = sentence[wordindex]

    features = {
        'isupper': word.isupper(),
        'islower': word.islower(),
        'istitle': word.istitle(),
        'isdigit': word.isdigit()
        # 'next word': sentence[wordindex + 1],
        # 'next word tag': labels[wordindex + 1],
        # 'tag': labels[wordindex]
    }
    return features
