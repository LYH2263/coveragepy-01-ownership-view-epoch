# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/coveragepy/coveragepy/blob/main/NOTICE.txt

"""View-finding generations and epoch-scoped framing memory.

Coverage.py keeps several planes separate:

* the ownership plane decides per file whether it is measured, and why
  (source / include / omit / third-party / stdlib / coverage.py itself);
* the memory plane remembers filename -> ownership decisions so the tracing
  hot path doesn't re-run the ownership decision on every call;
* the canonical-name plane resolves (possibly relative) file names through
  sys.path;
* the tracer stack plane keeps the file data already framed for live frames.

Import paths (sys.path) can change in the middle of a measurement session,
for example when a test runner manipulates sys.path.  A *view-finding epoch*
binds those planes to one import-path generation.  When sys.path changes:

1. the ownership matchers that depend on sys.path are recomputed and the
   epoch advances;
2. epoch-scoped framing memory is invalidated (or unbound, in isolated mode)
   while collected line data is preserved;
3. sys.path-dependent canonical resolutions are dropped;
4. tracer hot paths and already-framed stack state notice the epoch change
   and re-judge the next event even when the same ``co_filename`` re-enters.

Nothing here collapses ownership, memory, canonical naming, nesting and
epochs into a single switch: each plane keeps its own state and is driven by
the shared epoch counter defined here.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from coverage.files import advance_canonical_generation

if TYPE_CHECKING:
    from coverage.types import TFileDisposition

# A callback fired (with the new epoch) after a view-finding epoch advances.
AdvanceCallback = Callable[[int], None]


class ViewEpoch:
    """The current view-finding generation for one collector.

    Each collector owns its own ``ViewEpoch``, so nested collectors have
    independent generations even though they share process-wide sys.path and
    the process-wide canonical file-name cache.

    ``punch`` is the cheap hot-path entry point: it only does work when
    sys.path actually changed, preserving the "one ownership decision per file
    per generation" performance property.

    """

    def __init__(
        self,
        *,
        isolated: bool = False,
        on_advance: Sequence[AdvanceCallback] | None = None,
    ) -> None:
        self.isolated = isolated
        self.epoch = 0
        self._sys_path = list(sys.path)
        self._on_advance: list[AdvanceCallback] = list(on_advance or ())

    def __repr__(self) -> str:
        return f"<ViewEpoch epoch={self.epoch} isolated={self.isolated}>"

    def add_on_advance(self, callback: AdvanceCallback) -> None:
        """Register a callback to run after each epoch advancement."""
        self._on_advance.append(callback)

    def bind(self) -> int:
        """Bind a new framing decision to the current epoch."""
        return self.epoch

    def punch(self) -> int:
        """Notice any sys.path change, advancing the epoch if needed.

        Returns the epoch now in effect.  When sys.path is unchanged this is
        a single list comparison, so stable framing rules still cost nothing
        on the tracing hot path.

        """
        if sys.path != self._sys_path:
            self.advance()
        return self.epoch

    def advance(self) -> int:
        """Advance to a new view-finding epoch unconditionally."""
        self._sys_path = list(sys.path)
        self.epoch += 1

        # Relative canonical names are resolved through sys.path: invalidate
        # that process-wide plane before any plane re-decides anything.
        advance_canonical_generation()

        for callback in self._on_advance:
            callback(self.epoch)
        return self.epoch


class FramingMemory:
    """Epoch-scoped memory of filename -> ownership decisions.

    This is the memory plane shared between a collector and its tracers.  It
    deliberately stores ownership decisions only; collected line data lives
    in the collector's separate data buckets and is never cleared here.

    In non-isolated mode an epoch advancement drops the remembered decisions
    (the next encounter re-decides under the new rules).  In isolated mode the
    already-observed terminal decisions are retained and immutable: only their
    epoch binding is dropped, so a nested collector's later sys.path change
    cannot write through the outer collector's observed states.  New decisions
    are appended per epoch rather than replacing old ones.

    """

    def __init__(self, *, isolated: bool = False) -> None:
        self.isolated = isolated
        # The raw mapping is also exposed as ``should_trace_cache`` so that the
        # C tracer keeps working with its plain dict contract.
        self.raw: dict[str, TFileDisposition | None] = {}
        # Epoch that each raw entry is currently bound to.
        self._epochs: dict[str, int] = {}
        # Isolated mode only: the full, append-only history of observed
        # decisions: filename -> [(epoch, disposition), ...].
        self._history: dict[str, list[tuple[int, TFileDisposition | None]]] = {}

    def get(self, filename: str, epoch: int) -> TFileDisposition | None:
        """Return the decision bound to `filename` at `epoch`, if any.

        A missing entry and an unbound (stale) entry both mean "decide now".
        In isolated mode the history can still answer for an older epoch that
        is re-entered.

        """
        disposition = self.raw.get(filename)
        if disposition is None:
            return None
        if self._epochs.get(filename) == epoch:
            return disposition
        if self.isolated:
            for old_epoch, old_disp in reversed(self._history.get(filename, ())):
                if old_epoch == epoch:
                    return old_disp
        return None

    def put(
        self,
        filename: str,
        epoch: int,
        disposition: TFileDisposition | None,
    ) -> None:
        """Remember `disposition` for `filename` as decided at `epoch`."""
        if self.isolated:
            self._history.setdefault(filename, []).append((epoch, disposition))
        self.raw[filename] = disposition
        self._epochs[filename] = epoch

    def invalidate(self) -> None:
        """Unbind decisions after an epoch advancement.

        Isolated mode keeps the observed decisions (and their history) but
        unbinds them.  Non-isolated mode forgets the decisions entirely.  In
        neither case is collected line data touched.

        """
        self._epochs.clear()
        if not self.isolated:
            self.raw.clear()
            self._history.clear()

    def reset(self, raw: dict[str, TFileDisposition | None] | None = None) -> None:
        """Forget every decision for a brand-new measurement session."""
        if raw is not None:
            self.raw = raw
        else:
            self.raw.clear()
        self._epochs.clear()
        self._history.clear()

    def observed_decisions(
        self,
        filename: str,
    ) -> list[tuple[int, TFileDisposition | None]]:
        """Return the observed (epoch, decision) pairs for `filename`."""
        if self.isolated:
            return list(self._history.get(filename, ()))
        if filename in self.raw:
            return [(self._epochs.get(filename, -1), self.raw[filename])]
        return []
