from alibi_detect.utils.missing_optional_dependency import import_optional

fetch_detector, fetch_tf_model = import_optional('alibi_detect.utils.fetching.fetching',
                                                 names=['fetch_detector', 'fetch_tf_model'])

__all__ = ['fetch_tf_model', 'fetch_detector']
