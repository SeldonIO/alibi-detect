import non_existent_module  # noqa: F401


class MockedClassMissingRequiredDeps:
    def __init__(self):
        self.opt_dep = "opt_dep"


def mocked_function_missing_required_deps():
    pass


class MockedClassMissingMultipleRequiredDeps:
    def __init__(self):
        pass
