# mypy: disable-error-code="assignment, no-redef"

try:
    from pydantic.v1.fields import *  # noqa: F401, F403
except ImportError:
    from pydantic.fields import *  # noqa: F401, F403
