
# @author: Antriksh Agarwal
# Version 0: 4/13/2018

from __future__ import print_function
import numpy as np
from nltk.corpus import wordnet as wn
import logging
logging.basicConfig(filename='out1.log', format='%(asctime)s %(message)s')
logging.getLogger().setLevel(logging.INFO)

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


def getEntity(label):
    for entity in entities.keys():
        if entity in label:
            return entities[entity]

    return 0


def logger(message):
    logging.info(message)
    print(message)
