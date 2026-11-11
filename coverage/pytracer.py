# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/coveragepy/coveragepy/blob/main/NOTICE.txt

"""Raw data collector for coverage.py."""

from __future__ import annotations

import atexit
import dis
import itertools
import sys
import threading
from collections.abc import Callable
from types import FrameType, ModuleType
from typing import Any, cast

from coverage import env
from coverage.types import (
    TArc,
    TFileDisposition,
    TLineNo,
    Tracer,
    TShouldStartContextFn,
    TShouldTraceFn,
    TTraceData,
    TTraceFileData,
    TTraceFn,
    TWarnFn,
)

# I don't understand why, but if we use `cast(set[TLineNo], ...)` inside
# the _trace() function, we get some strange behavior on PyPy 3.10.
# Assigning these names here and using them below fixes the problem.
# See https://github.com/coveragepy/coveragepy/issues/1902
set_TLineNo = set[TLineNo]
set_TArc = set[TArc]


# We need the YIELD_VALUE opcode below, in a comparison-friendly form.
# PYVERSIONS: RESUME is new in CPython 3.11
RESUME = dis.opmap.get("RESUME")
RETURN_VALUE = dis.opmap["RETURN_VALUE"]
# PyPy attributes the exception-handler PUSH_EXC_INFO to the try/else line.
# That instruction is not the else body. Skip it so the else line stays missing.
PUSH_EXC_INFO = dis.opmap.get("PUSH_EXC_INFO")
if RESUME is None:
    YIELD_VALUE = dis.opmap["YIELD_VALUE"]
    YIELD_FROM = dis.opmap["YIELD_FROM"]
    YIELD_FROM_OFFSET = 0 if env.PYPY else 2
else:
    YIELD_VALUE = YIELD_FROM = YIELD_FROM_OFFSET = -1

# When running meta-coverage, this file can try to trace itself, which confuses
# everything.  Don't trace ourselves.

THIS_FILE = __file__.rstrip("co")


class PyTracer(Tracer):
    """Python implementation of the raw data tracer."""

    # Because of poor implementations of trace-function-manipulating tools,
    # the Python trace function must be kept very simple.  In particular, there
    # must be only one function ever set as the trace function, both through
    # sys.settrace, and as the return value from the trace function.  Put
    # another way, the trace function must always return itself.  It cannot
    # swap in other functions, or return None to avoid tracing a particular
    # frame.
    #
    # The trace manipulator that introduced this restriction is DecoratorTools,
    # which sets a trace function, and then later restores the pre-existing one
    # by calling sys.settrace with a function it found in the current frame.
    #
    # Systems that use DecoratorTools (or similar trace manipulations) must use
    # PyTracer to get accurate results.  The command-line --timid argument is
    # used to force the use of this tracer.

    tracer_ids = itertools.count()

    def __init__(self) -> None:
        # Which tracer are we?
        self.id = next(self.tracer_ids)

        # Attributes set from the collector:
        self.data: TTraceData
        self.trace_arcs = False
        self.should_trace: TShouldTraceFn
        self.should_trace_cache: dict[str, TFileDisposition | None]
        self.should_start_context: TShouldStartContextFn | None = None
        self.switch_context: Callable[[str | None], str | None] | None = None
        self.lock_data: Callable[[], None]
        self.unlock_data: Callable[[], None]
        self.warn: TWarnFn

        # View-finding epoch wiring, set from the collector.  When None (for
        # direct unit-test construction) the tracer falls back to the plain
        # should_trace_cache behavior.
        self.view_epoch: Any = None
        self.frame_memory: Any = None
        # The epoch our current cur_file_data/cur_file_name framing was bound
        # to.  A mismatch means even a same-filename re-entry must re-judge.
        self.framed_epoch = -1

        # The threading module to use, if any.
        self.threading: ModuleType | None = None

        self.cur_file_data: TTraceFileData | None = None
        self.last_line: TLineNo = 0
        self.cur_file_name: str | None = None
        self.context: str | None = None
        self.started_context = False

        # The data_stack parallels the Python call stack. Each entry is
        # information about an active frame, a five-element tuple:
        #   [0] The TTraceData for this frame's file. Could be None if we
        #           aren't tracing this frame.
        #   [1] The current file name for the frame. None if we aren't tracing
        #           this frame.
        #   [2] The last line number executed in this frame.
        #   [3] Boolean: did this frame start a new context?
        #   [4] The view-finding epoch this frame was framed under, so a
        #           restored caller frame can re-judge after path generations.
        self.data_stack: list[
            tuple[TTraceFileData | None, str | None, TLineNo, bool, int]
        ] = []
        self.thread: threading.Thread | None = None
        self.stopped = False
        self._activity = False

        self.in_atexit = False
        # On exit, self.in_atexit = True
        atexit.register(setattr, self, "in_atexit", True)

        # Cache a bound method on the instance, so that we don't have to
        # re-create a bound method object all the time.
        self._cached_bound_method_trace: TTraceFn = self._trace

    def __repr__(self) -> str:
        points = sum(len(v) for v in self.data.values())
        files = len(self.data)
        return f"<PyTracer at {id(self):#x}: {points} data points in {files} files>"

    def log(self, marker: str, *args: Any) -> None:
        """For hard-core logging of what this tracer is doing."""
        with open("/tmp/debug_trace.txt", "a", encoding="utf-8") as f:
            f.write(f"{marker} {self.id}[{len(self.data_stack)}]")
            if 0:  # if you want thread ids..
                f.write(  # type: ignore[unreachable]
                    f".{self.thread.ident:x}.{self.threading.current_thread().ident:x}"
                )
            f.write(" {}".format(" ".join(map(str, args))))
            if 0:  # if you want callers..
                f.write(" | ")  # type: ignore[unreachable]
                stack = " / ".join(
                    (fname or "???").rpartition("/")[-1]
                    for _, fname, _, _, _ in self.data_stack
                )
                f.write(stack)
            f.write("\n")

    def _frame_file(
        self,
        filename: str,
        frame: FrameType,
        epoch: int,
    ) -> None:
        """Bind the framing for `filename` to view-finding epoch `epoch`.

        Looks up the epoch-scoped ownership memory, re-running the ownership
        decision on a miss (which itself re-checks sys.path and may advance the
        epoch), then attaches the right data bucket to ``cur_file_data`` or
        disables line tracing for an excluded file.

        """
        self.cur_file_name = filename
        if self.frame_memory is not None:
            disp = self.frame_memory.get(filename, epoch)
            if disp is None:
                disp = self.should_trace(filename, frame)
                self.frame_memory.put(filename, epoch, disp)
            self.framed_epoch = epoch
        else:
            # Legacy path: no epoch wiring, speak the raw cache dict directly.
            disp = self.should_trace_cache.get(filename)
            if disp is None:
                disp = self.should_trace(filename, frame)
                self.should_trace_cache[filename] = disp

        self.cur_file_data = None
        if disp.trace:
            tracename = disp.source_filename
            assert tracename is not None
            self.lock_data()
            try:
                if tracename not in self.data:
                    self.data[tracename] = set()
            finally:
                self.unlock_data()
            self.cur_file_data = self.data[tracename]
        else:
            frame.f_trace_lines = False

    def _reframe_live_frame(self, frame: FrameType) -> None:
        """Re-judge the frame currently on top after an epoch advancement.

        Used from line/return events when sys.path changed while the frame was
        already live: the next recorded line or arc must follow the new
        ownership decision even though no new call event happened.  Already
        recorded lines stay in their buckets; invalidation never removes them.

        """
        assert self.view_epoch is not None
        epoch = self.view_epoch.punch()
        if epoch != self.framed_epoch:
            self._frame_file(frame.f_code.co_filename, frame, epoch)

    def _trace(
        self,
        frame: FrameType,
        event: str,
        arg: Any,  # pylint: disable=unused-argument
        lineno: TLineNo | None = None,  # pylint: disable=unused-argument
    ) -> TTraceFn | None:
        """The trace function passed to sys.settrace."""

        if THIS_FILE in frame.f_code.co_filename:
            return None

        # f = frame; code = f.f_code
        # self.log(":", f"{code.co_filename} {f.f_lineno} {code.co_name}()", event)

        if self.stopped and sys.gettrace() == self._cached_bound_method_trace:  # pylint: disable=comparison-with-callable
            # The PyTrace.stop() method has been called, possibly by another
            # thread, let's deactivate ourselves now.
            if 0:
                f = frame  # type: ignore[unreachable]
                self.log("---\nX", f.f_code.co_filename, f.f_lineno)
                while f:
                    self.log(">", f.f_code.co_filename, f.f_lineno, f.f_code.co_name, f.f_trace)
                    f = f.f_back
            sys.settrace(None)
            try:
                (
                    self.cur_file_data,
                    self.cur_file_name,
                    self.last_line,
                    self.started_context,
                    self.framed_epoch,
                ) = self.data_stack.pop()
            except IndexError:
                self.log(
                    "Empty stack!",
                    frame.f_code.co_filename,
                    frame.f_lineno,
                    frame.f_code.co_name,
                )
            return None

        # if event != "call" and frame.f_code.co_filename != self.cur_file_name:
        #     self.log("---\n*", frame.f_code.co_filename, self.cur_file_name, frame.f_lineno)

        if event == "call":
            # Should we start a new context?
            if self.should_start_context and self.context is None:
                context_maybe = self.should_start_context(frame)  # pylint: disable=not-callable
                if context_maybe is not None:
                    self.context = context_maybe
                    started_context = True
                    assert self.switch_context is not None
                    self.switch_context(self.context)  # pylint: disable=not-callable
                else:
                    started_context = False
            else:
                started_context = False
            self.started_context = started_context

            # Entering a new frame.  Decide if we should trace in this file.
            self._activity = True
            if self.view_epoch is not None:
                # The hot-path epoch check: even when the same co_filename
                # re-enters and no new context started, an advanced epoch must
                # force re-framing.  When nothing changed, punch() is just a
                # list comparison and the one-decision-per-file-per-epoch
                # property is preserved.
                epoch = self.view_epoch.punch()
                needs_framing = (
                    frame.f_code.co_filename != self.cur_file_name
                    or started_context
                    or epoch != self.framed_epoch
                )
            else:
                epoch = -1
                needs_framing = (
                    frame.f_code.co_filename != self.cur_file_name or started_context
                )

            self.data_stack.append(
                (
                    self.cur_file_data,
                    self.cur_file_name,
                    self.last_line,
                    started_context,
                    self.framed_epoch,
                ),
            )

            # Improve tracing performance: when calling a function, both caller
            # and callee are often within the same file. if that's the case, we
            # don't have to re-check whether to trace the corresponding
            # function (which is a little bit expensive since it involves
            # dictionary lookups). This optimization is only correct if we
            # didn't start a context, and the view-finding epoch hasn't moved.
            if needs_framing:
                self._frame_file(frame.f_code.co_filename, frame, epoch)
            elif not self.cur_file_data:
                frame.f_trace_lines = False

            # The call event is really a "start frame" event, and happens for
            # function calls and re-entering generators.  The f_lasti field is
            # -1 for calls, and a real offset for generators.  Use <0 as the
            # line number for calls, and the real line number for generators.
            if RESUME is not None:
                # The current opcode is guaranteed to be RESUME. The argument
                # determines what kind of resume it is.
                oparg = frame.f_code.co_code[frame.f_lasti + 1]
                real_call = (oparg == 0)  # fmt: skip
            else:
                real_call = (getattr(frame, "f_lasti", -1) < 0)  # fmt: skip
            if real_call:
                self.last_line = -frame.f_code.co_firstlineno
            else:
                self.last_line = frame.f_lineno

        elif event == "line":
            # Record an executed line.
            if (
                self.view_epoch is not None
                and self.framed_epoch != self.view_epoch.epoch
            ):
                # sys.path changed while this frame was on the stack: re-judge
                # before recording so new lines follow the new ownership rule.
                self._reframe_live_frame(frame)
            if self.cur_file_data is not None:
                if PUSH_EXC_INFO is not None:
                    lasti = frame.f_lasti
                    code = frame.f_code.co_code
                    if (
                        isinstance(lasti, int)
                        and 0 <= lasti < len(code)
                        and code[lasti] == PUSH_EXC_INFO
                    ):
                        return self._cached_bound_method_trace
                flineno: TLineNo = frame.f_lineno

                if self.trace_arcs:
                    cast(set_TArc, self.cur_file_data).add((self.last_line, flineno))
                else:
                    cast(set_TLineNo, self.cur_file_data).add(flineno)
                self.last_line = flineno

        elif event == "return":
            if (
                self.view_epoch is not None
                and self.framed_epoch != self.view_epoch.epoch
            ):
                # The epoch moved while this frame was live (for example the
                # frame itself rewrote sys.path): re-judge before the final
                # arc is recorded.
                self._reframe_live_frame(frame)
            if self.trace_arcs and self.cur_file_data:
                # Record an arc leaving the function, but beware that a
                # "return" event might just mean yielding from a generator.
                code = frame.f_code.co_code
                lasti = frame.f_lasti
                if RESUME is not None:
                    if len(code) == lasti + 2:
                        # A return from the end of a code object is a real return.
                        real_return = True
                    else:
                        # It is a real return if we aren't going to resume next.
                        if env.PYBEHAVIOR.lasti_is_yield:
                            lasti += 2
                        real_return = code[lasti] != RESUME
                else:
                    if code[lasti] == RETURN_VALUE:
                        real_return = True
                    elif code[lasti] == YIELD_VALUE:
                        real_return = False
                    elif len(code) <= lasti + YIELD_FROM_OFFSET:
                        real_return = True
                    elif code[lasti + YIELD_FROM_OFFSET] == YIELD_FROM:
                        real_return = False
                    else:
                        real_return = True
                if real_return:
                    # Python 3.14 can report a new line only on the return event.
                    if env.PYVERSION >= (3, 14) and frame.f_lineno != self.last_line:
                        cast(set_TArc, self.cur_file_data).add((self.last_line, frame.f_lineno))
                        self.last_line = frame.f_lineno
                    first = frame.f_code.co_firstlineno
                    cast(set_TArc, self.cur_file_data).add((self.last_line, -first))

            # Leaving this function, pop the filename stack.
            (
                self.cur_file_data,
                self.cur_file_name,
                self.last_line,
                self.started_context,
                restored_epoch,
            ) = self.data_stack.pop()
            self.framed_epoch = restored_epoch
            # Leaving a context?
            if self.started_context:
                assert self.switch_context is not None
                self.context = None
                self.switch_context(None)  # pylint: disable=not-callable
            # The restored caller frame was framed under an older generation:
            # re-frame it under the current epoch so its subsequent events
            # follow the new rules instead of the pre-change cur_file_data.
            if (
                self.view_epoch is not None
                and self.cur_file_name is not None
                and restored_epoch != self.view_epoch.epoch
            ):
                self._frame_file(
                    self.cur_file_name,
                    frame.f_back,
                    self.view_epoch.punch(),
                )

        return self._cached_bound_method_trace

    def start(self) -> TTraceFn:
        """Start this Tracer.

        Return a Python function suitable for use with sys.settrace().

        """
        self.stopped = False
        if self.threading:
            if self.thread is None:
                self.thread = self.threading.current_thread()

        sys.settrace(self._cached_bound_method_trace)
        return self._cached_bound_method_trace

    def stop(self) -> None:
        """Stop this Tracer."""
        # Get the active tracer callback before setting the stop flag to be
        # able to detect if the tracer was changed prior to stopping it.
        tf = sys.gettrace()

        # Set the stop flag. The actual call to sys.settrace(None) will happen
        # in the self._trace callback itself to make sure to call it from the
        # right thread.
        self.stopped = True

        if self.threading:
            assert self.thread is not None
            if self.thread.ident != self.threading.current_thread().ident:
                # Called on a different thread than started us: we can't unhook
                # ourselves, but we've set the flag that we should stop, so we
                # won't do any more tracing.
                # self.log("~", "stopping on different threads")
                return

        # PyPy clears the trace function before running atexit functions,
        # so don't warn if we are in atexit on PyPy and the trace function
        # has changed to None.  Metacoverage also messes this up, so don't
        # warn if we are measuring ourselves.
        suppress_warning = (env.PYPY and self.in_atexit and tf is None) or env.METACOV
        if self.warn and not suppress_warning:
            if tf != self._cached_bound_method_trace:  # pylint: disable=comparison-with-callable
                self.warn(
                    "Trace function changed, data is likely wrong: "
                    + f"{tf!r} != {self._cached_bound_method_trace!r}",
                    slug="trace-changed",
                )

    def activity(self) -> bool:
        """Has there been any activity?"""
        return self._activity

    def reset_activity(self) -> None:
        """Reset the activity() flag."""
        self._activity = False

    def get_stats(self) -> dict[str, int] | None:
        """Return a dictionary of statistics, or None."""
        return None
