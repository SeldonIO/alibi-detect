# ###### CHANGED in TF-TFP-TORCH OPTIONAL DEPS PR ####################
# ###### these comments should be removed prior to PR merge. #########
# ####################################################################
#
# 2. Moved `saving` from `/utils/saving.py` to here: `/utils/saving/`.
# `save_detector` and `load_detector` are `import_optional`'ed in
# `/utils/saving/__init__.py`.
# TODO: config-driven-detectors moves this module out of utils. So
#  will require refactoring!


from alibi_detect.utils.missing_optional_dependency import import_optional

save_detector, load_detector = import_optional(
    'alibi_detect.utils.saving.saving',
    names=['save_detector', 'load_detector']
)
