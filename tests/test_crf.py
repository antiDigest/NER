# test_crf.py

# @author: Antriksh Agarwal
# Version 0: 4/13/2018

import time


def test_crf_import():

    import ConditionalRandomField
    from ConditionalRandomField import ConditionalRandomField


def test_crf_train():
    import ConditionalRandomField
    from ConditionalRandomField import ConditionalRandomField
    # from ConditionalRandomField import Chain
    from DataSet import DataSet

    d = DataSet(FILE='demo/sample.csv')
    crf = ConditionalRandomField(d)
    crf.train()
