# mypy: disable-error-code="assignment, no-redef"

try:
    from pydantic.v1 import *  # noqa: F401, F403
except ImportError:
    from pydantic import *  # noqa: F401, F403
