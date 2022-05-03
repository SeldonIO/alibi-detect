from alibi_detect.utils.random import set_seed, get_seed, fixed_seed


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
    tmp_seed = seed + 42
    with fixed_seed(tmp_seed):
        # Check seed inside of context manager
        assert get_seed() == tmp_seed

    # Check seeds were reset upon exit of context manager
    assert get_seed() == seed
