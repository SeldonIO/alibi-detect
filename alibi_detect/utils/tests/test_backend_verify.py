import pytest
from alibi_detect.utils.missing_optional_dependency import ERROR_TYPES
from alibi_detect.utils.frameworks import BackendValidator, HAS_BACKEND


class TestImportOptional:
    """Test the import_optional function."""

    def setup_method(self):
        # mock missing dependency error for non existent module
        ERROR_TYPES['backend_B'] = 'optional-deps'
        ERROR_TYPES['backend_E'] = 'backend_E'
        ERROR_TYPES['backend_D'] = 'backend_D'
        HAS_BACKEND['backend_A'] = True
        HAS_BACKEND['backend_B'] = False
        HAS_BACKEND['backend_C'] = True
        HAS_BACKEND['backend_D'] = False
        HAS_BACKEND['backend_E'] = False

    def teardown_method(self):
        # remove mock missing dependency error for other tests
        del HAS_BACKEND['backend_A']
        del HAS_BACKEND['backend_B']
        del HAS_BACKEND['backend_C']
        del HAS_BACKEND['backend_D']
        del HAS_BACKEND['backend_E']
        del ERROR_TYPES['backend_B']
        del ERROR_TYPES['backend_E']
        del ERROR_TYPES['backend_D']

    def test_backend_verifier_error_msg(self):
        options = {
            'backend_1': ['backend_A', 'backend_B'],
            'backend_2': ['backend_C'],
            'backend_3': ['backend_D', 'backend_E'],
        }
        with pytest.raises(NotImplementedError) as err:
            BackendValidator(backend_options=options, construct_name='test').verify_backend('test')
        assert str(err.value) == ('test backend not implemented. Use one of `backend_1`, '
                                  '`backend_2` and `backend_3` instead.')

        with pytest.raises(ImportError) as err:
            BackendValidator(backend_options=options, construct_name='test').verify_backend('backend_1')
        assert str(err.value) == ('`backend_B` not installed. Cannot initialize and run test with backend_1 backend.'
                                  ' The necessary missing dependencies can be installed using '
                                  '`pip install alibi-detect[optional-deps]`.')

        BackendValidator(backend_options=options, construct_name='test').verify_backend('backend_2')

        with pytest.raises(ImportError) as err:
            BackendValidator(backend_options=options, construct_name='test').verify_backend('backend_3')
        assert 'backend_D' in str(err.value).split('.')[0]
        assert 'backend_E' in str(err.value).split('.')[0]
        assert 'backend_3' in str(err.value)
        assert 'backend_D' in str(err.value).split('pip install alibi-detect')[1]
        assert 'backend_E' in str(err.value).split('pip install alibi-detect')[1]
