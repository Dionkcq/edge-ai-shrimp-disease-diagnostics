"""Optional, local, additive advice generation.

Nothing in this package may be consulted for the screening decision itself: the
decision and its cited guidance are produced entirely by
:mod:`shrimp_screening.policy` and :mod:`shrimp_screening.guidance` before
anything here runs. This package only ever expands an already-decided,
already-cited result into a longer, farmer-facing explanation, and only when a
local Ollama server is configured and reachable.
"""

from __future__ import annotations
