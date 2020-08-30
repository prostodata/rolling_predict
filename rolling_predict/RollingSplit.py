"""
The :mod:`rolling_predict.RollingSplit` module includes classes and functions
to split time series in a running mode.
"""

import numpy as np

class RollingSplit:
    """Splits the data using running mode. Splits dataset consecutively
    by moving through the dataset. For each new consecutive position, the
    train set include the specified number of previous (past) rows, and
    the test set includes the specified number of next (future) rows.

    Parameters
    ----------
    train_size : int
        Size of the training set.
        This number of rows will be retrieved from the past
    train_size : int, default=1
        Size of the test set.
        This number of rows will be retrieved from the future
    start : int, default=0
        The first split. Every next split will be made after `train_size` rows
    end : int
        The last possible split
    n_splits : int
        Maximum number of train/test datasets produced
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import KFold
    >>> rs = RollingSplit(train_size=5, test_size=2)
    >>> rs.get_n_splits(10)
    2
    >>> for train_index, test_index in rs.split(10):
    ...     print("TRAIN: {} TEST: {}".format(train_index, test_index))
    TRAIN: [0 1 2 3 4] TEST: [5 6]
    TRAIN: [2 3 4 5 6] TEST: [7 8]
    """
    def __init__(self, train_size, test_size=1, start=None, end=None, n_splits=None):
        self.train_size = train_size
        self.test_size = test_size

        self.start = start
        self.end = end
        self.n_splits = n_splits

        if not self.train_size:
            raise ValueError("Train size must be specified.")

    def split(self, size):
        """Generate indices to a train and test data sets rolling through the dataset of the specified size.
        Parameters
        ----------
        size : int
            Size of the dataset
        Yields
        ------
        train : ndarray
            The training set indices for that rolling split.
        test : ndarray
            The testing set indices for that rolling split.
        """

        self._restore_missing_and_validate(size)

        #
        # Iterate and return index arrays
        #

        indices = np.arange(size)

        test_starts = [self.start + x*self.test_size for x in range(self.n_splits)]
        for test_start in test_starts:
            yield (
                # Train set consists of previous (past) indexes
                indices[test_start - self.train_size:test_start],
                # Test set consists of next (future) indexes
                indices[test_start:test_start + self.test_size]
            )

    def get_n_splits(self, size):
        """Returns the number of splitting iterations in the cross-validator
        Parameters
        ----------
        size : int
            Size of the dataset
        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the rolling split
            through the dataset of this size
        """
        self._restore_missing_and_validate(size)
        return self.n_splits

    def _restore_missing_and_validate(self, size):
        #
        # Derive missing parameters
        #
        if not self.start:
            self.start = self.train_size

        if not self.end:
            self.end = size

        # How many full (optionally, also one last partial) test sizes are between start and end
        n_splits_max = (self.end - self.start) // self.test_size
        if not self.n_splits:
            self.n_splits = n_splits_max
        else:
            # Ensure that there are enough data and decrease if necessary
            self.n_splits = min(self.n_splits, n_splits_max)

        #
        # Validation
        #

        # Train size cannot be less that the start
        if self.start < self.train_size:
            raise ValueError("Size is larger than the start. Not enough past data for the (first) training set. Either increase start or decrease ")
