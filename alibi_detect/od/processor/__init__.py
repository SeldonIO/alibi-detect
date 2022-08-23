import imp
from alibi_detect.utils.missing_optional_dependency import import_optional
from alibi_detect.od.processor.base import BaseProcessor

ParallelProcessor = import_optional('alibi_detect.od.processor.parallel', names=['ParallelProcessor'])