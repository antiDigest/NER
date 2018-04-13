# test_dataset.py

# @author: Antriksh Agarwal
# Version 0: 4/13/2018


def test_dataset():

	from DataSet import DataSet

	start = time.time()
    d = DataSet()
    print time.time() - start
    for row in d.iterate():
        print row
        break

