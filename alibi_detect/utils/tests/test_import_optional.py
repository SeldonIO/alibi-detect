import pytest

from alibi_detect.utils.missing_optional_dependency import import_optional, MissingDependency, ERROR_TYPES


class TestImportOptional:
    """Test the import_optional function."""

    def setup_method(self):
        # mock missing dependency error for non existent module
        ERROR_TYPES['non_existent_module'] = 'optional-deps'

    def teardown_method(self):
        # remove mock missing dependency error for other tests
        del ERROR_TYPES['non_existent_module']

    def test_import_optional_module(self):
        """Test import_optional correctly imports installed module."""
        requests = import_optional('requests')
        assert requests.__version__

    def test_import_optional_names(self):
        """Test import_optional correctly imports names from installed module."""
        from requests import get, post
        get2, post2 = import_optional('requests', names=['get', 'post'])
        assert get2 == get
        assert get2.__name__ == 'get'
        assert post2 == post
        assert post2.__name__ == 'post'

    def test_import_optional_module_missing(self):
        """Test import_optional correctly replaces module that doesn't exist with MissingDependency."""
        package = import_optional('alibi_detect.utils.tests.mocked_opt_dep')
        assert isinstance(package, MissingDependency)
        with pytest.raises(ImportError) as err:
            package.__version__  # noqa
        assert 'alibi_detect.utils.tests.mocked_opt_dep' in str(err.value)
        assert 'pip install alibi-detect[optional-deps]' in str(err.value)

        with pytest.raises(ImportError) as err:
            package(0, 'test')  # noqa
        assert 'alibi_detect.utils.tests.mocked_opt_dep' in str(err.value)
        assert 'pip install alibi-detect[optional-deps]' in str(err.value)

    def test_import_optional_names_missing(self):
        """Test import_optional correctly replaces names from module that doesn't exist with MissingDependencies."""
        MockedClassMissingRequiredDeps, mocked_function_missing_required_deps = import_optional(
            'alibi_detect.utils.tests.mocked_opt_dep',
            names=['MockedClassMissingRequiredDeps', 'mocked_function_missing_required_deps'])
        assert isinstance(MockedClassMissingRequiredDeps, MissingDependency)
        with pytest.raises(ImportError) as err:
            MockedClassMissingRequiredDeps.__version__  # noqa
        assert 'MockedClassMissingRequiredDeps' in str(err.value)
        assert 'pip install alibi-detect[optional-deps]' in str(err.value)

        assert isinstance(mocked_function_missing_required_deps, MissingDependency)
        with pytest.raises(ImportError) as err:
            mocked_function_missing_required_deps.__version__  # noqa
        assert 'mocked_function_missing_required_deps' in str(err.value)
        assert 'pip install alibi-detect[optional-deps]' in str(err.value)
