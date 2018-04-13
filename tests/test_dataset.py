# test_dataset.py

# @author: Antriksh Agarwal
# Version 0: 4/13/2018

import time


def test_dataset_import_test():

    from DataSet import DataSet


def test_data_to_records():
    from DataSet import DataSet

    d = DataSet(FILE='data/kaggle/sample.csv')
    r = None
    for row in d.iterate():
        r = row
        break

    for row in d.to_records():
        assert r[1] == row[1].split(';')
        break
