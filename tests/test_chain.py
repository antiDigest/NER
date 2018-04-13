# test_chain.py

# @author: Antriksh Agarwal
# Version 0: 4/13/2018

import time


def test_chain_import():

    from ConditionalRandomField.ConditionalRandomField import Chain

    d = DataSet()
    for row in d.iterate():
        print row
        break


def test_data_to_records():
    from DataSet import DataSet

    d = DataSet()
    r = None
    for row in d.iterate():
        r = row
        break

    for row in d.to_records():
        assert r[1] == row[1].split(';')
        break
