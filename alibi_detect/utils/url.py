from typing import Union, List
from urllib.parse import urljoin, quote_plus


def _join_url(base: str, parts: Union[str, List[str]]) -> str:
    """
    Constructs a full (“absolute”) URL by combining a “base URL” (base) with additional relative URL parts.
    The behaviour is similar to os.path.join() on linux, but also behaves consistently on Windows.

    Parameters
    ----------
    base
        The base URL, e.g. `'https://mysite.com/'`.
    parts
        Part to append, or list of parts to append e.g. `['/dir1/', 'dir2', 'dir3']`.

    Returns
    -------
    The joined url e.g. `https://mysite.com/dir1/dir2/dir3`.
    """
    parts = [parts] if isinstance(parts, str) else parts
    if len(parts) == 0:
        raise TypeError("The `parts` argument must contain at least one item.")
    url = urljoin(base + "/", "/".join(quote_plus(part.strip(r"\/"), safe="/") for part in parts))
    return url
