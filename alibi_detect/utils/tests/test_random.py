from alibi_detect.utils._random import set_seed, get_seed, fixed_seed
import numpy as np
import tensorflow as tf
import torch


def test_set_get_seed(seed):
    """
    Tests the set_seed and get_seed fuctions.
    """
    # Check initial seed within test is the one set by pytest-randomly
    current_seed = get_seed()
    assert current_seed == seed

    # Set another seed and check
    new_seed = seed + 42
    set_seed(new_seed)
    current_seed = get_seed()
    assert current_seed == new_seed


def test_fixed_seed(seed):
    """
    Tests the fixed_seed context manager.
    """
    n = 5  # Length of random number sequences

    nums0 = []
    tmp_seed = seed + 42
    with fixed_seed(tmp_seed):
        # Generate a sequence of random numbers
        for i in range(n):
            nums0.append(np.random.normal([1]))
            nums0.append(tf.random.normal([1]))
            nums0.append(torch.normal(torch.tensor([1.0])))

        # Check seed unchanged after RNG calls
        assert get_seed() == tmp_seed

    # Generate another sequence of random numbers with same seed, and check equal
    nums1 = []
    tmp_seed = seed + 42
    with fixed_seed(tmp_seed):
        for i in range(n):
            nums1.append(np.random.normal([1]))
            nums1.append(tf.random.normal([1]))
            nums1.append(torch.normal(torch.tensor([1.0])))
    assert nums0 == nums1

    # Generate another sequence of random numbers with different seed, and check not equal
    nums2 = []
    tmp_seed = seed + 99
    with fixed_seed(tmp_seed):
        for i in range(n):
            nums2.append(np.random.normal([1]))
            nums2.append(tf.random.normal([1]))
            nums2.append(torch.normal(torch.tensor([1.0])))
    assert nums1 != nums2

    # Check seeds were reset upon exit of context managers
    assert get_seed() == seed
