from importlib.metadata import version as _version

try:
    __version__ = _version("fylearn")
except Exception:
    __version__ = "unknown"
