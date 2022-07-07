# ###### CHANGED in TF-TFP-TORCH OPTIONAL DEPS PR ####################
# ###### these comments should be removed prior to PR merge. #########
# ####################################################################
#
# 1. fetching.py is dependent on tensorflow optional dependencies but
# _join_url is used in datasets so this function has been seperated
# into `alibi_detect.utils.fetching.url.py`.


from alibi_detect.utils.missing_optional_dependency import import_optional

fetch_detector, fetch_tf_model = import_optional('alibi_detect.utils.fetching.fetching',
                                                 names=['fetch_detector', 'fetch_tf_model'])

__all__ = ['fetch_tf_model', 'fetch_detector']
