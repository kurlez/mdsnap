"""
Command-line interface for converting Markdown documents into shareable image cards.

This package exposes a :func:`main` function which orchestrates argument parsing
and delegates rendering work to :mod:`scripts.md_to_cards`.
"""

from .cli import main

__all__ = ["main"]

