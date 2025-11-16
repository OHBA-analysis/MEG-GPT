from importlib.metadata import PackageNotFoundError, version

# Setup the version
try:
    __version__ = version("EphysGPT")
except PackageNotFoundError:
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError
