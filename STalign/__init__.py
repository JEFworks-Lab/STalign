__version__ = '2.0.0'

# The functions of the module STalign.STalign are also those of the package, so that
# `import STalign; STalign.LDDMM(...)` works with an installed STalign as it does when the
# tutorials import STalign.py from the source tree (their "OPTION B"), and as
# `from STalign import STalign; STalign.LDDMM(...)` does.
from .STalign import *  # noqa: E402,F401,F403
