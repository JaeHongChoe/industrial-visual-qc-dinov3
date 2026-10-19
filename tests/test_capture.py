from pathlib import Path

from patch_qc.capture import parse_standard_capture, rank_extra_captures


def test_parse_standard_capture_preserves_family_and_index() -> None:
    condition = parse_standard_capture("_A_Photo-L1-03.jpg")

    assert condition is not None
    assert condition.family == "A"
    assert condition.view_index == 3
    assert condition.condition_id == "A_L1-03"
    assert condition.is_standard is True


def test_parse_standard_capture_accepts_m1_lowercase_family() -> None:
    condition = parse_standard_capture("M1-c_Photo-L1-12.jpg")

    assert condition is not None
    assert condition.family == "C"
    assert condition.view_index == 12
    assert condition.condition_id == "C_L1-12"


def test_parse_standard_capture_accepts_collision_suffix() -> None:
    condition = parse_standard_capture("M1-a_Photo-L1-03_0001.jpg")

    assert condition is not None
    assert condition.condition_id == "A_L1-03"


def test_parse_standard_capture_rejects_unresolved_extra_capture() -> None:
    assert parse_standard_capture("M1-c__039.jpg") is None


def test_rank_extra_captures_is_deterministic_and_natural() -> None:
    paths = [Path("M1-c__103.jpg"), Path("M1-c__99.jpg"), Path("M1-c__100.jpg")]

    ranked = rank_extra_captures(paths)

    assert ranked[Path("M1-c__99.jpg")].condition_id == "extra_1"
    assert ranked[Path("M1-c__100.jpg")].condition_id == "extra_2"
    assert ranked[Path("M1-c__103.jpg")].condition_id == "extra_3"
