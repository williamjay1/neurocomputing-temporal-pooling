"""Relocate immutable experiment scripts and historical path bindings in memory.

The producer files and evidence JSONs are never edited.  This adapter substitutes
only the Path constructor visible to those scripts and the loader used by their
explicit imports.  No scientific function, assertion, array or hash is replaced.
"""
from __future__ import annotations

import builtins
import hashlib
import importlib
import importlib.abc
import importlib.util
import pathlib
import sys
import types

ORIGINAL_ROOT = "D:/MLWork/Neurocomputing_TailTransfer"
NativePath = type(pathlib.Path())


def digest(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


class LegacyRuntime:
    def __init__(self, root, output):
        self.root = pathlib.Path(root).resolve()
        self.output = pathlib.Path(output).resolve()
        self.loaded = {}
        self.path_mappings = {}
        runtime = self

        class RelocatedPath(NativePath):
            @staticmethod
            def relocated_args(args):
                if args and isinstance(args[0], (str, pathlib.PurePath)):
                    text = str(args[0]).replace("\\", "/")
                    old = ORIGINAL_ROOT.lower()
                    if text.lower() == old or text.lower().startswith(old + "/"):
                        suffix = text[len(ORIGINAL_ROOT):].lstrip("/")
                        if "/site-packages/" in suffix:
                            # Only the frozen third-party N-HiTS source binding
                            # points into the historical virtual environment.
                            package_file = suffix.split("/site-packages/", 1)[1]
                            if not package_file.startswith("neuralforecast/"):
                                raise ValueError("Undeclared dependency relocation: " + text)
                            from importlib.metadata import distribution
                            replacement = pathlib.Path(distribution("neuralforecast").locate_file(package_file))
                        else:
                            replacement = runtime.root / suffix
                        runtime.path_mappings[text] = str(replacement)
                        args = (str(replacement),) + args[1:]
                return args

            def __new__(cls, *args, **kwargs):
                return super().__new__(cls, *cls.relocated_args(args), **kwargs)

            def __init__(self, *args, **kwargs):
                # Python3.12 pathlib parses arguments in __init__, whereas
                # older versions parsed in __new__.
                super().__init__(*self.relocated_args(args), **kwargs)

        self.Path = RelocatedPath
        self.pathlib_proxy = types.ModuleType("pathlib")
        self.pathlib_proxy.__dict__.update(pathlib.__dict__)
        self.pathlib_proxy.Path = self.Path
        self.util_proxy = types.ModuleType("importlib.util")
        self.util_proxy.__dict__.update(importlib.util.__dict__)
        self.util_proxy.spec_from_file_location = self.spec_from_file_location
        self.importlib_proxy = types.ModuleType("importlib")
        self.importlib_proxy.__dict__.update(importlib.__dict__)
        self.importlib_proxy.util = self.util_proxy

    def _import(self, name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name == "pathlib":
            return self.pathlib_proxy
        if level == 0 and name in ("importlib", "importlib.util"):
            return self.util_proxy if name == "importlib.util" and fromlist else self.importlib_proxy
        return builtins.__import__(name, globals, locals, fromlist, level)

    def spec_from_file_location(self, name, location, *args, **kwargs):
        path = self.Path(location)
        spec = importlib.util.spec_from_file_location(name, path, *args, **kwargs)
        if path.resolve().parent == self.root / "scripts":
            spec.loader = _ImmutableScriptLoader(self, path)
        return spec

    def load(self, filename, name=None):
        path = self.root / "scripts" / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        name = name or "release_" + path.stem
        spec = self.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    def bind_output(self, mod, group):
        """Redirect generated results, preserving all read-only input roots."""
        mod.OUT = self.Path(self.output / group)
        mod.OUT.mkdir(parents=True, exist_ok=True)
        return mod


class _ImmutableScriptLoader(importlib.abc.Loader):
    def __init__(self, runtime, path):
        self.runtime, self.path = runtime, path

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        before = digest(self.path)
        module.__dict__["__builtins__"] = dict(vars(builtins), __import__=self.runtime._import)
        code = compile(self.path.read_bytes(), str(self.path), "exec")
        exec(code, module.__dict__)
        after = digest(self.path)
        if before != after:
            raise AssertionError("Immutable script changed during import: " + str(self.path))
        self.runtime.loaded[self.path.relative_to(self.runtime.root).as_posix()] = after
