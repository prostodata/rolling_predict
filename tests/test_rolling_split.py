import numpy as np

from rolling_predict import *

def test_split():
    # Train using 5 past samples. Predict using 2 future samples
    # Rolling step is 2. There 2 full train-test cycles
    rs = RollingSplit(train_size=5, test_size=2)

    n_splits = rs.get_n_splits(10)
    assert n_splits == 2

    splits = list(rs.split(10))

    assert len(splits) == 2

    assert np.array_equal(splits[0][0], [0,1,2,3,4])
    assert np.array_equal(splits[0][1], [5,6])

    assert np.array_equal(splits[1][0], [2,3,4,5,6])
    assert np.array_equal(splits[1][1], [7,8])

    for train_index, test_index in rs.split(10):
        print("TRAIN: {} TEST: {}".format(train_index, test_index))

    pass
