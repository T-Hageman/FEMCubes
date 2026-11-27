"""Tests for the command-line options shared by the example scripts."""

from pathlib import Path

import pytest

from femcubes.core.utils.example_utils import default_output_dir, parse_example_arguments


def test_defaults_without_cases_write_next_to_the_script(tmp_path):
    script = tmp_path / "examples" / "demo.py"
    args = parse_example_arguments(str(script), "demo", argv=[])

    assert args.case is None
    assert not args.quick
    assert args.plots
    assert args.output_dir == script.resolve().parent / "results" / "demo"


def test_cases_default_to_the_first_and_get_their_own_folder(tmp_path):
    script = tmp_path / "demo.py"
    args = parse_example_arguments(str(script), "demo", cases=["a", "b"], argv=[])

    assert args.case == "a"
    assert args.output_dir == script.resolve().parent / "results" / "demo" / "a"


def test_options_are_parsed(tmp_path):
    out = tmp_path / "out"
    args = parse_example_arguments(str(tmp_path / "demo.py"), "demo", cases=["a", "b"], default_case="b",
                                   argv=["--quick", "--no-plots", "--output-dir", str(out)])

    assert (args.case, args.quick, args.plots, args.output_dir) == ("b", True, False, out.resolve())
    assert parse_example_arguments("demo.py", "demo", cases=["a", "b"], argv=["--case", "a"]).case == "a"


def test_relative_output_dir_is_resolved(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = parse_example_arguments("demo.py", "demo", argv=["--output-dir", "out"])

    assert args.output_dir == Path(tmp_path, "out").resolve()


@pytest.mark.parametrize("argv", [["--case", "c"], ["--case", "a"]])
def test_invalid_or_unavailable_case_is_rejected(argv):
    cases = ["a", "b"] if argv[1] == "c" else None
    with pytest.raises(SystemExit):
        parse_example_arguments("demo.py", "demo", cases=cases, argv=argv)


@pytest.mark.parametrize("cases, default_case", [([], None), (["a"], "b"), (None, "a")])
def test_invalid_case_definitions_are_rejected(cases, default_case):
    with pytest.raises(ValueError):
        parse_example_arguments("demo.py", "demo", cases=cases, default_case=default_case, argv=[])


def test_default_output_dir_matches_the_parsed_default(tmp_path):
    script = str(tmp_path / "demo.py")
    assert default_output_dir(script, "a") == parse_example_arguments(script, "demo", cases=["a"], argv=[]).output_dir
    assert default_output_dir(script) == tmp_path.resolve() / "results" / "demo"


def test_script_options_are_added_with_their_types():
    options = {"rate": 1e-10, "n_notches": 1, "rheology": "VE", "plane_stress": False}
    args = parse_example_arguments("demo.py", "demo", options=options, argv=[])
    assert args.options == options

    args = parse_example_arguments("demo.py", "demo", options=options,
                                   argv=["--rate", "1e-11", "--n-notches", "3", "--rheology", "LE", "--plane-stress"])
    assert args.options == {"rate": 1e-11, "n_notches": 3, "rheology": "LE", "plane_stress": True}
    assert isinstance(args.options["n_notches"], int)


@pytest.mark.parametrize("options", [{"flag": True}, {"values": [1, 2]}])
def test_invalid_script_options_are_rejected(options):
    with pytest.raises(ValueError):
        parse_example_arguments("demo.py", "demo", options=options, argv=[])
