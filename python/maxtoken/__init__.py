"""The pre-rename import name: ``maxtoken`` is maxMLX.

``import maxtoken.x.y`` returns the very module ``maxmlx.x.y`` (one module
object under two names, not a second copy with its own state), so scripts
and pickles written against the old name keep working.
"""

import importlib
import importlib.abc
import importlib.util
import sys

import maxmlx

_OLD, _NEW = "maxtoken", "maxmlx"


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, target: str):
        self._target = target

    def create_module(self, spec):
        return importlib.import_module(self._target)

    def exec_module(self, module):
        pass


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(_OLD + "."):
            return None
        new = _NEW + fullname[len(_OLD):]
        if importlib.util.find_spec(new) is None:
            return None
        return importlib.util.spec_from_loader(fullname, _AliasLoader(new))


if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())
sys.modules[_OLD] = maxmlx
