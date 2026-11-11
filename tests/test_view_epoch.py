# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/coveragepy/coveragepy/blob/main/NOTICE.txt

"""Tests for view-finding epochs and nested-collector isolation.

These tests exercise, all in one process and one measurement session:

* the distinct ownership planes (source / include / omit / third-party /
  coverage.py itself) with their different reasons;
* the divergence between report-time trimming and measurement-time ownership;
* the bidirectional flips ("was excluded, now collected" and "was collected,
  now excluded") after a real sys.path rewrite, with the flips happening on
  re-entry of the *same* ``co_filename`` where the PyTracer hot path would
  normally skip the ownership decision entirely;
* the one-decision-per-file-per-generation property while framing rules are
  stable;
* nested collectors where the inner collector rewrites sys.path and the outer
  collector re-frames on resume instead of keeping its pre-nesting memory;
* isolation mode in which already-bound generations and observed terminal
  states cannot be written through by the inner collector;
* get_data() settlement consistency after epoch invalidation.

Every measurement here is forced onto the Python tracer (``timid=True``), as
required: the sys.monitoring core must not be used to stand in for it.
"""

from __future__ import annotations

import os
import os.path
import sys

import pytest

from coverage import Coverage, files
from coverage.exceptions import NoDataError
from coverage.framing import FramingMemory, ViewEpoch
from coverage.pytracer import PyTracer
from tests.coveragetest import CoverageTest

FLIPMOD_SRC = """\
MARKER = {marker!r}
def work(n):
    total = {base}
    for i in range(n):
        total += i
    return total
def driver():
    return work({arg})
"""

FLIPOUT_V1 = """\
PHASE = 'v1'
def outer_work():
    x = 1
    y = 2
    return x + y
"""

FLIPOUT_V2 = """\
PHASE = 'v2'
def outer_work():
    x = 1
    y = 2
    z = 3
    extra = 4
    return x + y + z + extra
"""

INNER_SRC = """\
def inner_work():
    a = 10
    b = 20
    return a + b
"""

# A module that rewrites sys.path *inside a live frame* and then calls another
# function in the same module: the worker() call re-enters the identical
# co_filename without a new context, exactly the spot where the tracing hot
# path used to skip every kind of re-judgment.
HOTFLIP_SRC = """\
MARKER = {marker!r}
def worker(n):
    total = {base}
    for i in range(n):
        total += i
    return total

def flip_then_call(flip_hook, n):
    flip_hook()
    return worker(n)
"""


class ViewEpochTest(CoverageTest):
    """Same-process tests of view-finding generations and isolation."""

    def setUp(self) -> None:
        super().setUp()
        self._saved_sys_path = list(sys.path)
        self.addCleanup(self._restore_sys_path)
        # The canonical file-name cache is process-global; reset it so tests
        # never observe each other's sys.path generations.
        self.addCleanup(files.set_relative_directory)

    def _restore_sys_path(self) -> None:
        sys.path[:] = self._saved_sys_path

    def _make_generation_dirs(self) -> None:
        """Create v1/ and v2/ trees carrying same-named modules."""
        self.make_file(
            os.path.join("v1", "flipmod.py"),
            FLIPMOD_SRC.format(marker="v1", base=0, arg=5),
        )
        self.make_file(
            os.path.join("v2", "flipmod.py"),
            FLIPMOD_SRC.format(marker="v2", base=1, arg=6),
        )
        self.make_file(os.path.join("v1", "flipout.py"), FLIPOUT_V1)
        self.make_file(os.path.join("v2", "flipout.py"), FLIPOUT_V2)
        self.make_file(os.path.join("v2", "innermod.py"), INNER_SRC)
        self.make_file(
            os.path.join("v1", "hotflip.py"),
            HOTFLIP_SRC.format(marker="v1", base=0),
        )
        self.make_file(
            os.path.join("v2", "hotflip.py"),
            HOTFLIP_SRC.format(marker="v2", base=1),
        )
        self.make_file(
            os.path.join("v1", "hotout.py"),
            HOTFLIP_SRC.format(marker="v1", base=0),
        )
        self.make_file(
            os.path.join("v2", "hotout.py"),
            HOTFLIP_SRC.format(marker="v2", base=1),
        )

    def _use_path_dir(self, dirname: str) -> None:
        """Make `dirname` (a relative entry) the only generation path."""
        for old in ("v1", "v2"):
            while old in sys.path:
                sys.path.remove(old)
        sys.path.insert(0, dirname)

    def _load_and_run(self, module_name: str, runs: int = 5) -> None:
        """Resolve a module through the real sys.path, then execute it.

        The source is compiled with the bare module name as co_filename, so
        re-entry after a sys.path flip keeps the *same* co_filename even
        though canonical resolution now points at a different directory.
        """
        path = None
        for dirname in sys.path:
            candidate = os.path.join(dirname, module_name)
            if os.path.exists(candidate):
                path = candidate
                break
        assert path is not None, f"{module_name} not found on sys.path"
        with open(path, encoding="utf-8") as f:
            source = f.read()
        code = compile(source, module_name, "exec")
        namespace = {"__name__": module_name[:-3], "__file__": module_name}
        exec(code, namespace)
        entry_points = {
            "flipmod.py": "driver",
            "flipout.py": "outer_work",
            "innermod.py": "inner_work",
        }
        for _ in range(runs):
            namespace[entry_points[module_name]]()

    def _exec_hotflip(self, module_name: str, flip_hook, n: int = 3):
        """Run flip_then_call from a module resolved via the real sys.path.

        The hook runs inside a live frame of `module_name`; the subsequent
        same-module worker call re-enters the identical co_filename without a
        new dynamic context.
        """
        path = None
        for dirname in sys.path:
            candidate = os.path.join(dirname, module_name)
            if os.path.exists(candidate):
                path = candidate
                break
        assert path is not None, f"{module_name} not found on sys.path"
        with open(path, encoding="utf-8") as f:
            source = f.read()
        code = compile(source, module_name, "exec")
        namespace = {"__name__": module_name[:-3], "__file__": module_name}
        exec(code, namespace)
        return namespace["flip_then_call"](flip_hook, n)

    def _scoped_measured(self, data) -> set[str]:
        """Measured files belonging to this test's v1/v2 generation trees."""
        roots = (os.path.abspath("v1"), os.path.abspath("v2"))
        return {
            f for f in data.measured_files()
            if os.path.abspath(f).startswith(roots)
        }

    def _counting_coverage(self, **kwargs: object) -> Coverage:
        """A timid (PyTracer) Coverage recording every ownership decision."""
        cov = Coverage(config_file=False, timid=True, data_file=None, **kwargs)
        decisions: list[tuple[str, int, bool, str]] = []
        original = cov._should_trace

        def counting(filename, frame=None):
            disp = original(filename, frame)
            # The holder has already advanced (if needed) by the time the
            # ownership decision runs, so this is the generation the decision
            # is effectively bound to.
            epoch = cov._collector.view_epoch.epoch if cov._collector is not None else -1
            decisions.append((filename, epoch, disp.trace, disp.reason))
            return disp

        cov._should_trace = counting  # type: ignore[method-assign]
        cov.view_decisions = decisions  # type: ignore[attr-defined]
        return cov

    def _assert_pytracer(self, cov: Coverage) -> None:
        assert cov._core is not None
        assert cov._core.tracer_class is PyTracer, (
            "acceptance requires the PyTracer core, "
            f"got {cov._core.tracer_class.__name__}"
        )

    def _decisions_for(
        self,
        cov: Coverage,
        co_filename: str,
    ) -> list[tuple[int, bool, str]]:
        return [
            (epoch, trace, reason)
            for name, epoch, trace, reason in cov.view_decisions
            if name == co_filename
        ]

    def test_ownership_planes_keep_distinct_reasons(self) -> None:
        # The ownership layer must not collapse source/include/omit/
        # third-party/coverage.py into a single on/off switch.
        self.make_file(os.path.join("srcdir", "a.py"), "a = 1\n")
        self.make_file(os.path.join("otherdir", "b.py"), "b = 2\n")
        self.make_file("c.py", "c = 3\n")
        self.make_file("d.py", "d = 4\n")

        # A venv-shaped third-party area on sys.path: pyvenv.cfg sits in an
        # ancestor of the importable site-packages directory.
        self.make_file(os.path.join("vendor", "pyvenv.cfg"), "home = /usr/bin\n")
        self.make_file(
            os.path.join("vendor", "lib", "site-packages", "vendored.py"),
            "v = 1\n",
        )
        vendor_site_packages = os.path.abspath(
            os.path.join("vendor", "lib", "site-packages"),
        )
        sys.path.insert(0, vendor_site_packages)

        source_cov = Coverage(
            config_file=False, timid=True, data_file=None,
            source=[os.path.abspath("srcdir")],
        )
        source_cov.start()
        self._assert_pytracer(source_cov)
        try:
            inside = source_cov._inorout.should_trace(os.path.abspath(os.path.join("srcdir", "a.py")))
            outside = source_cov._inorout.should_trace(os.path.abspath(os.path.join("otherdir", "b.py")))
        finally:
            source_cov.stop()
        assert inside.trace is True
        assert outside.trace is False
        source_outside_reason = outside.reason

        include_cov = Coverage(
            config_file=False, timid=True, data_file=None, include=["c.py"],
        )
        include_cov.start()
        try:
            include_miss = include_cov._inorout.should_trace(os.path.abspath("otherdir/b.py"))
            include_hit = include_cov._inorout.should_trace(os.path.abspath("c.py"))
        finally:
            include_cov.stop()
        assert include_miss.trace is False
        assert include_hit.trace is True

        omit_cov = Coverage(
            config_file=False, timid=True, data_file=None,
            omit=[os.path.abspath("d.py")],
        )
        omit_cov.start()
        try:
            omit_disp = omit_cov._inorout.should_trace(os.path.abspath("d.py"))
            # The real coverage.py package's own source, not a same-named
            # file in the temp directory.
            import coverage as coverage_package
            coverage_disp = omit_cov._inorout.should_trace(
                os.path.abspath(coverage_package.__file__),
            )
            vendor_disp = omit_cov._inorout.should_trace(
                os.path.abspath(
                    os.path.join("vendor", "lib", "site-packages", "vendored.py"),
                ),
            )
        finally:
            omit_cov.stop()

        assert omit_disp.trace is False
        assert coverage_disp.trace is False
        assert vendor_disp.trace is False

        reasons = {
            "source_outside": source_outside_reason,
            "include_outside": include_miss.reason,
            "omit": omit_disp.reason,
            "coverage_itself": coverage_disp.reason,
            "third_party": vendor_disp.reason,
        }
        # Every plane speaks with its own reason; nothing is collapsed.
        assert len(set(reasons.values())) == len(reasons), reasons
        assert "source" in reasons["source_outside"]
        assert "include" in reasons["include_outside"]
        assert "omit" in reasons["omit"]
        assert "coverage.py" in reasons["coverage_itself"]
        assert "third-party" in reasons["third_party"]

    def test_report_trimming_never_rewrites_measurement(self) -> None:
        self.make_file("f1.py", "x = 1\n")
        self.make_file("f2.py", "y = 2\n")
        cov = Coverage(config_file=False, timid=True, data_file=None)
        cov.start()
        self._assert_pytracer(cov)
        import_local = __import__("coverage.misc", fromlist=["import_local_file"]).import_local_file
        import_local("f1")
        import_local("f2")
        cov.stop()

        data = cov.get_data()
        measured = {
            f for f in data.measured_files()
            if os.path.basename(f) in {"f1.py", "f2.py"}
        }
        assert measured == {os.path.abspath("f1.py"), os.path.abspath("f2.py")}

        # Report-only omit hides f1 from the report, but must not touch the
        # measured data produced during measurement.
        report = self.get_report(cov, morfs=[], omit=[os.path.abspath("f1.py")])
        assert "f1.py" not in report
        assert "f2.py" in report
        assert os.path.abspath("f1.py") in set(cov.get_data().measured_files())

        # A report include cannot conjure an unmeasured file into the data.
        with pytest.raises(NoDataError):
            self.get_report(cov, morfs=[], include=["never_measured.py"])
        assert measured == {
            f for f in cov.get_data().measured_files()
            if os.path.basename(f) in {"f1.py", "f2.py"}
        }

    def test_bidirectional_flip_on_same_cofilename_hot_path(self) -> None:
        self._make_generation_dirs()
        v1_flipmod = os.path.abspath(os.path.join("v1", "flipmod.py"))
        v2_flipmod = os.path.abspath(os.path.join("v2", "flipmod.py"))
        v1_flipout = os.path.abspath(os.path.join("v1", "flipout.py"))
        v2_flipout = os.path.abspath(os.path.join("v2", "flipout.py"))

        cov = self._counting_coverage(omit=[v1_flipmod, v2_flipout])
        self._use_path_dir("v1")
        cov.start()
        self._assert_pytracer(cov)
        tracer = cov._collector.tracers[0]

        # Generation 0: flipmod is omitted, flipout is collected.  Repeated
        # calls exercise the same-co_filename hot path over and over.
        self._load_and_run("flipmod.py", runs=5)
        self._load_and_run("flipout.py", runs=5)
        assert cov._collector.view_epoch.epoch == 0
        data_after_g0 = cov.get_data()
        assert v1_flipmod not in set(data_after_g0.measured_files())
        flipout_lines_g0 = set(data_after_g0.lines(v1_flipout))
        assert flipout_lines_g0

        # Real in-process import path rewrite.  No new Coverage instance:
        # the same measurement session must observe the flip.
        self._use_path_dir("v2")
        self._load_and_run("flipmod.py", runs=5)
        self._load_and_run("flipout.py", runs=5)

        # The epoch and the process canonical generation must have advanced.
        assert cov._collector.view_epoch.epoch == 1
        assert files.canonical_generation() >= 1
        # The tracer's live framing moved to the new generation while frames
        # were active; after every frame returns the stack drains back to its
        # baseline, so assert on the generation stamped on the decisions.
        assert tracer.framed_epoch in (-1, 1)
        cov.stop()

        data = cov.get_data()
        measured = self._scoped_measured(data)

        # Forward flip: the same co_filename "flipmod.py" was re-judged and is
        # now collected from its new canonical location.
        assert v1_flipmod not in measured
        assert v2_flipmod in measured
        assert data.lines(v2_flipmod)

        # Reverse flip: flipout was collected at v1 but is omitted at v2.  The
        # old lines survive (invalidation never deletes data), but no new
        # bucket or lines appear for the excluded generation.
        assert v1_flipout in measured
        assert v2_flipout not in measured
        assert set(data.lines(v1_flipout)) == flipout_lines_g0

        # Both flips happened on re-entry of the identical co_filename string,
        # precisely where the hot path would previously skip any re-judgment.
        flipmod_decisions = self._decisions_for(cov, "flipmod.py")
        flipout_decisions = self._decisions_for(cov, "flipout.py")
        assert [(epoch, trace) for epoch, trace, _ in flipmod_decisions] == [
            (0, False),
            (1, True),
        ]
        assert [(epoch, trace) for epoch, trace, _ in flipout_decisions] == [
            (0, True),
            (1, False),
        ]
        # One decision per file per generation only, despite 10 executions.
        assert len(flipmod_decisions) == 2
        assert len(flipout_decisions) == 2

    def test_flip_inside_live_frame_on_same_cofilename_reentry(self) -> None:
        # The strict case: sys.path is rewritten *inside* an already-traced
        # frame, and the next event that could notice the change is a call to
        # another function carrying the identical co_filename, with no new
        # context and no intervening different-filename frame.  Both the
        # excluded->included and included->excluded directions must flip.
        self._make_generation_dirs()
        v1_hotflip = os.path.abspath(os.path.join("v1", "hotflip.py"))
        v2_hotflip = os.path.abspath(os.path.join("v2", "hotflip.py"))
        v1_hotout = os.path.abspath(os.path.join("v1", "hotout.py"))
        v2_hotout = os.path.abspath(os.path.join("v2", "hotout.py"))

        cov = self._counting_coverage(omit=[v1_hotflip, v2_hotout])
        self._use_path_dir("v1")
        cov.start()
        self._assert_pytracer(cov)

        # Generation 0 with no-op hooks: hotflip is omitted, hotout collected.
        for _ in range(3):
            self._exec_hotflip("hotflip.py", lambda: None)
            self._exec_hotflip("hotout.py", lambda: None)
        g0 = cov.get_data()
        hotout_lines_g0 = set(g0.lines(v1_hotout))
        assert hotout_lines_g0
        assert v1_hotflip not in self._scoped_measured(g0)

        def real_flip() -> None:
            self._use_path_dir("v2")

        # The flip happens within each module's own live frame; the following
        # same-co_filename worker() call is where re-judgment must occur.
        self._exec_hotflip("hotflip.py", real_flip)
        self._exec_hotflip("hotout.py", lambda: None)
        cov.stop()

        data = cov.get_data()
        measured = self._scoped_measured(data)
        # Excluded -> included on same co_filename re-entry.
        assert v1_hotflip not in measured
        assert v2_hotflip in measured
        assert data.lines(v2_hotflip)
        # Included -> excluded on same co_filename re-entry: old lines stay,
        # no new bucket is ever created for the excluded generation.
        assert v1_hotout in measured
        assert v2_hotout not in measured
        assert set(data.lines(v1_hotout)) == hotout_lines_g0

        hotflip_decisions = self._decisions_for(cov, "hotflip.py")
        hotout_decisions = self._decisions_for(cov, "hotout.py")
        assert [(e, t) for e, t, _ in hotflip_decisions] == [(0, False), (1, True)]
        assert [(e, t) for e, t, _ in hotout_decisions] == [(0, True), (1, False)]

    def test_stable_rules_decide_each_cofilename_once(self) -> None:
        self._make_generation_dirs()
        cov = self._counting_coverage()
        self._use_path_dir("v1")
        cov.start()
        self._assert_pytracer(cov)
        self._load_and_run("flipmod.py", runs=20)
        assert cov._collector.view_epoch.epoch == 0
        cov.stop()

        decisions = self._decisions_for(cov, "flipmod.py")
        assert len(decisions) == 1, decisions
        assert decisions[0][0] == 0  # bound to the initial generation
        assert decisions[0][1] is True

    def test_canonical_relative_cache_follows_path_generation(self) -> None:
        self._make_generation_dirs()
        v1 = os.path.abspath(os.path.join("v1", "flipmod.py"))
        v2 = os.path.abspath(os.path.join("v2", "flipmod.py"))
        absolute = os.path.abspath(os.path.join("v1", "flipout.py"))

        self._use_path_dir("v1")
        holder = ViewEpoch()
        assert files.canonical_filename("flipmod.py") == v1
        # An absolute name has no sys.path dependence.
        assert files.canonical_filename(absolute) == absolute

        # Rewrite sys.path and advance the generation, exactly as a collector
        # would do when it notices the change.
        self._use_path_dir("v2")
        holder.advance()
        assert files.canonical_filename("flipmod.py") == v2
        # Absolute entries are not invalidated by path generations.
        assert files.canonical_filename(absolute) == absolute

    def test_nested_collector_outer_reframes_on_resume(self) -> None:
        self._make_generation_dirs()
        v1_flipmod = os.path.abspath(os.path.join("v1", "flipmod.py"))
        v2_flipmod = os.path.abspath(os.path.join("v2", "flipmod.py"))
        v2_innermod = os.path.abspath(os.path.join("v2", "innermod.py"))

        # Outer session: flipmod omitted while v1 is on sys.path.
        outer = self._counting_coverage(omit=[v1_flipmod])
        self._use_path_dir("v1")
        outer.start()
        self._assert_pytracer(outer)
        self._load_and_run("flipmod.py", runs=3)
        assert outer._collector.view_epoch.epoch == 0

        # A genuinely nested second collector in the same process: its start
        # pauses the outer collector, its stop resumes it.
        inner = Coverage(config_file=False, timid=True, data_file=None)
        inner.start()
        try:
            self._use_path_dir("v2")
            self._load_and_run("flipmod.py", runs=3)
            self._load_and_run("innermod.py", runs=3)
        finally:
            inner.stop()

        # The outer collector has resumed: it must punch its epoch, notice the
        # inner path rewrite, and re-frame instead of reusing old memory.
        assert outer._collector.view_epoch.epoch >= 1
        self._load_and_run("flipmod.py", runs=3)
        outer.stop()

        outer_data = outer.get_data()
        outer_measured = self._scoped_measured(outer_data)
        # The outer's post-resume re-judgment collects the now-included file.
        assert v1_flipmod not in outer_measured
        assert v2_flipmod in outer_measured
        # Data buckets stay isolated: the inner's own module never leaks out.
        assert v2_innermod not in outer_measured

        inner_data = inner.get_data()
        inner_measured = self._scoped_measured(inner_data)
        assert v2_innermod in inner_measured
        assert v2_flipmod in inner_measured

        # The outer re-decided the same co_filename after resume rather than
        # silently keeping its pre-nesting "omit" decision.
        flipmod_decisions = self._decisions_for(outer, "flipmod.py")
        assert [(epoch, trace) for epoch, trace, _ in flipmod_decisions] == [
            (0, False),
            (1, True),
        ]

    def test_syspath_change_recomputes_ownership_matchers(self) -> None:
        # The ownership plane's sys.path-dependent matchers (third-party areas
        # and source packages inside them) must be recomputed when the epoch
        # advances: the same absolute file flips between third-party excluded
        # and traceable without anyone reconstructing the Coverage object.
        self.make_file(os.path.join("vendorenv", "pyvenv.cfg"), "home = /usr/bin\n")
        venmod = os.path.abspath(
            os.path.join("vendorenv", "lib", "site-packages", "venmod.py"),
        )
        self.make_file(
            os.path.join("vendorenv", "lib", "site-packages", "venmod.py"),
            "v = 1\n",
        )
        site_packages = os.path.dirname(venmod)
        sys.path.insert(0, site_packages)

        cov = Coverage(config_file=False, timid=True, data_file=None)
        cov.start()
        self._assert_pytracer(cov)
        holder = cov._collector.view_epoch
        try:
            first = cov._inorout.should_trace(venmod)
            assert first.trace is False
            assert "third-party" in first.reason
            assert holder.epoch == 0

            # Rewrite sys.path out from under the running session, then let
            # the next ownership decision punch the epoch.
            sys.path.remove(site_packages)
            second = cov._inorout.should_trace(venmod)
            assert holder.epoch == 1
            assert second.trace is True
            assert second.reason == ""

            # And back: re-adding the venv-shaped area excludes it again after
            # another recomputation.
            sys.path.insert(0, site_packages)
            third = cov._inorout.should_trace(venmod)
            assert holder.epoch == 2
            assert third.trace is False
            assert "third-party" in third.reason
        finally:
            cov.stop()

    def test_resume_advances_epoch_before_tracers_restart(self) -> None:
        # White-box ordering contract for nested resume: the view-finding
        # epoch (and therefore matcher recomputation, memory invalidation and
        # canonical invalidation) must advance inside resume(), before any
        # tracer is restarted, rather than lazily on the first later event.
        self._make_generation_dirs()
        cov = Coverage(config_file=False, timid=True, data_file=None)
        self._use_path_dir("v1")
        cov.start()
        collector = cov._collector
        holder = collector.view_epoch
        assert holder.epoch == 0

        events: list[tuple[str, int]] = []
        holder.add_on_advance(lambda epoch: events.append(("advance", epoch)))
        original_start = collector.tracers[0].start

        def spy_start() -> None:
            events.append(("tracer_start", holder.epoch))
            original_start()

        collector.tracers[0].start = spy_start  # type: ignore[method-assign]

        # Emulate the nested-collector pause/resume window.
        collector.pause()
        self._use_path_dir("v2")
        collector.resume()

        assert events, "resume must observe the path generation"
        assert events[0][0] == "advance"
        assert holder.epoch == 1
        # The tracer only restarts after the epoch has advanced.
        assert ("tracer_start", 1) in events
        cov.stop()

    def test_isolation_observed_states_are_not_written_through(self) -> None:
        self._make_generation_dirs()
        v1_flipmod = os.path.abspath(os.path.join("v1", "flipmod.py"))
        v2_flipmod = os.path.abspath(os.path.join("v2", "flipmod.py"))
        v2_innermod = os.path.abspath(os.path.join("v2", "innermod.py"))

        outer = Coverage(
            config_file=False, timid=True, data_file=None,
            view_isolation=True, omit=[v1_flipmod],
        )
        # Pin the first generation's import paths before starting, so the
        # first observed decision binds epoch 0.
        self._use_path_dir("v1")

        # Isolation forces the Python tracer.
        outer.start()
        self._assert_pytracer(outer)
        assert outer._collector.view_isolation is True
        memory = outer._collector.frame_memory
        assert memory.isolated is True

        self._load_and_run("flipmod.py", runs=3)
        observed = memory.observed_decisions("flipmod.py")
        assert len(observed) == 1
        epoch0_disp = observed[0][1]
        assert epoch0_disp is not None
        assert epoch0_disp.trace is False
        assert "omit" in epoch0_disp.reason
        outer_data_before = {
            f: set(outer.get_data().lines(f)) for f in outer.get_data().measured_files()
        }

        # Inner collector rewrites paths, fills the canonical cache and runs
        # its own modules while the outer collector is paused.
        inner = Coverage(config_file=False, timid=True, data_file=None)
        inner.start()
        try:
            self._use_path_dir("v2")
            self._load_and_run("innermod.py", runs=2)
            self._load_and_run("flipmod.py", runs=2)
            # The inner's relative resolution must not poison the outer: the
            # process generation moved and stale relative entries are dropped.
            assert files.canonical_filename("flipmod.py") == v2_flipmod
        finally:
            inner.stop()

        # Write-through protection: the outer's epoch-0 terminal decision is
        # still the same object with the same reason, and its measured data is
        # byte-for-byte untouched by the inner collector.
        retained = memory.observed_decisions("flipmod.py")
        assert retained[0][0] == 0
        assert retained[0][1] is epoch0_disp
        assert retained[0][1].reason == epoch0_disp.reason
        outer_data_after = {
            f: set(outer.get_data().lines(f)) for f in outer.get_data().measured_files()
        }
        assert outer_data_after == outer_data_before
        assert v2_innermod not in set(outer.get_data().measured_files())

        # New observations after resume bind the new generation and append
        # instead of overwriting the old terminal state.
        assert outer._collector.view_epoch.epoch >= 1
        self._load_and_run("flipmod.py", runs=3)
        history = memory.observed_decisions("flipmod.py")
        assert [epoch for epoch, _ in history] == [0, outer._collector.view_epoch.epoch]
        new_disp = history[-1][1]
        assert new_disp is not epoch0_disp
        assert new_disp.trace is True
        outer.stop()

        measured = set(outer.get_data().measured_files())
        assert v2_flipmod in measured
        assert v1_flipmod not in measured

    def test_non_isolated_mode_invalidates_memory_but_keeps_data(self) -> None:
        self._make_generation_dirs()
        v1_flipout = os.path.abspath(os.path.join("v1", "flipout.py"))
        cov = self._counting_coverage()
        self._use_path_dir("v1")
        cov.start()
        self._assert_pytracer(cov)
        self._load_and_run("flipout.py", runs=2)
        memory = cov._collector.frame_memory
        assert memory.isolated is False
        assert "flipout.py" in memory.raw
        lines_before = set(cov.get_data().lines(v1_flipout))

        self._use_path_dir("v2")
        assert cov._collector.view_epoch.punch() >= 1
        # Non-isolated mode drops the decisions themselves...
        assert "flipout.py" not in memory.raw
        # ...but never the collected line data.
        assert set(cov.get_data().lines(v1_flipout)) == lines_before
        cov.stop()

    def test_get_data_settlement_matches_rejudged_execution(self) -> None:
        self._make_generation_dirs()
        v1_flipmod = os.path.abspath(os.path.join("v1", "flipmod.py"))
        v2_flipmod = os.path.abspath(os.path.join("v2", "flipmod.py"))
        v1_flipout = os.path.abspath(os.path.join("v1", "flipout.py"))
        v2_flipout = os.path.abspath(os.path.join("v2", "flipout.py"))

        cov = self._counting_coverage(omit=[v1_flipmod, v2_flipout])
        self._use_path_dir("v1")
        cov.start()
        self._load_and_run("flipmod.py", runs=2)
        self._load_and_run("flipout.py", runs=2)
        frozen_lines = set(cov.get_data().lines(v1_flipout))

        self._use_path_dir("v2")
        self._load_and_run("flipmod.py", runs=2)
        self._load_and_run("flipout.py", runs=2)
        cov.stop()

        # get_data() itself must settle the view-finding epoch before
        # finalizing the measured file set.
        assert cov._collector.settled_epoch == 0
        data = cov.get_data()
        settled_epoch = cov._collector.settled_epoch
        assert settled_epoch == cov._collector.view_epoch.epoch
        assert settled_epoch >= 1
        measured = self._scoped_measured(data)
        assert measured == {v2_flipmod, v1_flipout}
        assert data.lines(v2_flipmod)
        assert set(data.lines(v1_flipout)) == frozen_lines
        assert v1_flipmod not in measured
        assert v2_flipout not in measured


class FramingMemoryUnitTest(CoverageTest):
    """Focused tests for the epoch-scoped memory plane semantics."""

    run_in_temp_dir = False

    class _DummyDisposition:
        def __init__(self, trace: bool, reason: str) -> None:
            self.trace = trace
            self.reason = reason

    def test_stable_epoch_serves_cached_decisions(self) -> None:
        memory = FramingMemory()
        disp = self._DummyDisposition(True, "")
        memory.put("a.py", 0, disp)
        assert memory.get("a.py", 0) is disp
        assert memory.get("a.py", 1) is None

    def test_non_isolated_invalidate_forgets_decisions(self) -> None:
        memory = FramingMemory(isolated=False)
        memory.put("a.py", 0, self._DummyDisposition(True, ""))
        memory.invalidate()
        assert memory.raw == {}
        assert memory.get("a.py", 1) is None

    def test_isolated_invalidate_keeps_observed_history(self) -> None:
        memory = FramingMemory(isolated=True)
        first = self._DummyDisposition(False, "omit")
        second = self._DummyDisposition(True, "")
        memory.put("a.py", 0, first)
        memory.invalidate()
        # The observed terminal state survives and remains attributable.
        assert memory.observed_decisions("a.py") == [(0, first)]
        assert memory.get("a.py", 1) is None
        memory.put("a.py", 1, second)
        assert memory.get("a.py", 1) is second
        # History is append-only: the old decision object is never mutated.
        assert memory.observed_decisions("a.py") == [(0, first), (1, second)]

    def test_view_epoch_punch_advances_only_on_real_path_change(self) -> None:
        holder = ViewEpoch()
        assert holder.punch() == 0
        assert holder.punch() == 0
        saved = list(sys.path)
        try:
            sys.path.insert(0, "definitely_not_on_path_before")
            assert holder.punch() == 1
            assert holder.bind() == 1
            assert holder.punch() == 1
        finally:
            sys.path[:] = saved
