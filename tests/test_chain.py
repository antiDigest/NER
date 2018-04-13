# test_chain.py

# @author: Antriksh Agarwal
# Version 0: 4/13/2018

import time


def test_chain_import():

    import ConditionalRandomField
    from ConditionalRandomField import ConditionalRandomField
    # from ConditionalRandomField import Chain


def test_chain_probability():
    import ConditionalRandomField
    from ConditionalRandomField import ConditionalRandomField
    # from ConditionalRandomField import Chain
    from DataSet import DataSet

    d = DataSet(FILE='demo/sample.csv')
    crf = ConditionalRandomField(d)
    chains = crf.getChains()
    for chain in chains:
        chain.probability(crf.weights)
