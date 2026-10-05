"""Backend adapters for benchmark circuits."""

from .merlin import run_merlin
from .external import run_external

__all__ = ["run_merlin", "run_external"]
