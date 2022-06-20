# ###### CHANGED in TF-TFP-TORCH OPTIONAL DEPS PR ####################
# ###### these comments should be removed prior to PR merge. #########
# ####################################################################
#
# 1. The fetching file has been refactored to `/url.py`, `/fetch_tf_models.py` and
# `/fetching.py`.
# 2. url file contains _join_url which is used in datasets. It's been moved there so that it can be imported without
#   causing missing dep errors w.r.t. tensorflow and pytorch.
# 3. Can't recall why /fetch_tf_models and /fetching have been refactored! TODO: Find out or refactor!


from alibi_detect.utils.missing_optional_dependency import import_optional

fetch_detector = import_optional('alibi_detect.utils.fetching.fetching', names=['fetch_detector'])
fetch_tf_model = import_optional('alibi_detect.utils.fetching.fetch_tf_models', names=['fetch_tf_model'])

__all__ = ['fetch_tf_model', 'fetch_detector']
