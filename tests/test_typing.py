# Adapted from https://github.com/numpy/numpy/blob/main/numpy/typing/tests/test_typing.py

import os
import re
from collections import defaultdict
from collections.abc import Iterator

import pytest
from mypy import api

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PASS_DIR = os.path.join(DATA_DIR, "pass")
FAIL_DIR = os.path.join(DATA_DIR, "fail")
MYPY_INI = os.path.join(DATA_DIR, "mypy.ini")

#: A dictionary with file names as keys and lists of the mypy stdout as values.
#: To-be populated by `run_mypy`.
OUTPUT_MYPY: defaultdict[str, list[str]] = defaultdict(list)


def _key_func(key: str) -> str:
    """Split at the first occurrence of the ``:`` character.

    Windows drive-letters (*e.g.* ``C:``) are ignored herein.
    """
    drive, tail = os.path.splitdrive(key)
    return os.path.join(drive, tail.split(":", 1)[0])


def _strip_filename(msg: str) -> tuple[int, str]:
    """Strip the filename and line number from a mypy message."""
    _, tail = os.path.splitdrive(msg)
    _, lineno, msg = tail.split(":", 2)
    return int(lineno), msg.strip()


@pytest.fixture(scope="module", autouse=True)
def run_mypy() -> None:
    split_pattern = re.compile(r"(\s+)?\^(\~+)?")
    for directory in (PASS_DIR, FAIL_DIR):
        # Run mypy
        stdout, stderr, exit_code = api.run(["--config-file", MYPY_INI, directory])
        if stderr:
            pytest.fail(f"Unexpected mypy standard error\n\n{stderr}")
        elif exit_code not in {0, 1}:
            pytest.fail(f"Unexpected mypy exit code: {exit_code}\n\n{stdout}")

        str_concat = ""
        filename: str | None = None
        for i in stdout.split("\n"):
            if "note:" in i:
                continue
            if filename is None:
                filename = _key_func(i)

            str_concat += f"{i}\n"
            if split_pattern.match(i) is not None:
                OUTPUT_MYPY[filename].append(str_concat)
                str_concat = ""
                filename = None


def get_test_cases(directory: str) -> Iterator:
    for root, _, files in os.walk(directory):
        for fname in files:
            short_fname, ext = os.path.splitext(fname)
            if ext in (".pyi", ".py"):
                fullpath = os.path.join(root, fname)
                yield pytest.param(fullpath, id=short_fname)


@pytest.mark.parametrize("path", get_test_cases(PASS_DIR))
def test_success(path) -> None:
    # Alias `OUTPUT_MYPY` so that it appears in the local namespace
    output_mypy = OUTPUT_MYPY
    if path in output_mypy:
        msg = "Unexpected mypy output\n\n"
        msg += "\n".join(_strip_filename(v)[1] for v in output_mypy[path])
        raise AssertionError(msg)


@pytest.mark.parametrize("path", get_test_cases(FAIL_DIR))
def test_fail(path: str) -> None:
    __tracebackhide__ = True

    with open(path) as fin:
        lines = fin.readlines()

    errors = defaultdict(lambda: "")

    output_mypy = OUTPUT_MYPY
    assert path in output_mypy

    for error_line in output_mypy[path]:
        lineno, error_line = _strip_filename(error_line)
        errors[lineno] += f"{error_line}\n"

    for i, line in enumerate(lines):
        lineno = i + 1
        if line.startswith("#") or (" E:" not in line and lineno not in errors):
            continue

        target_line = lines[lineno - 1]
        if "# E:" in target_line:
            expression, _, marker = target_line.partition("  # E: ")
            expected_error = errors[lineno].strip()
            marker = marker.strip()
            _test_fail(path, expression, marker, expected_error, lineno)
        else:
            pytest.fail(f"Unexpected mypy output at line {lineno}\n\n{errors[lineno]}")


_FAIL_MSG1 = """Extra error at line {}

Expression: {}
Extra error: {!r}
"""

_FAIL_MSG2 = """Error mismatch at line {}

Expression: {}
Expected error: {}
Observed error: {!r}
"""


def _test_fail(
    path: str,
    expression: str,
    error: str,
    expected_error: None | str,
    lineno: int,
) -> None:
    if expected_error is None:
        raise AssertionError(_FAIL_MSG1.format(lineno, expression, error))
    elif error not in expected_error:
        raise AssertionError(
            _FAIL_MSG2.format(lineno, expression, expected_error, error)
        )
