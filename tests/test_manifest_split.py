from pathlib import Path

from patch_qc.manifest import scan_ok_tree
from patch_qc.split import split_by_group


def _touch_capture(root: Path, date: str, product: str, object_id: str) -> None:
    capture = root / date / product / object_id
    capture.mkdir(parents=True)
    for name in ("_A_Photo-L1-01.jpg", "_B_Photo-L1-02.jpg", "M1-c__037.jpg"):
        (capture / name).touch()


def test_scan_ok_tree_filters_dates_and_classifies_extra_rank(tmp_path: Path) -> None:
    _touch_capture(tmp_path, "2026.04.30", "OLD", "G001")
    _touch_capture(tmp_path, "2026.05.01", "PRODUCT_A", "G001")
    _touch_capture(tmp_path, "20260502", "PRODUCT_A", "G002")

    records = scan_ok_tree(tmp_path, start_date="2026-05-01")

    assert len(records) == 2
    record = records[0]
    assert record.date == "2026-05-01"
    assert record.group_id == "PRODUCT_A::G001"
    assert [image.condition.condition_id for image in record.images] == [
        "A_L1-01",
        "B_L1-02",
        "extra_1",
    ]
    assert records[1].date == "2026-05-02"
    assert records[1].group_id == "PRODUCT_A::G002"


def test_split_by_group_never_leaks_repeated_unit_across_dates(tmp_path: Path) -> None:
    for date in ("2026.05.01", "2026.06.01"):
        _touch_capture(tmp_path, date, "PRODUCT_A", "G001")
    for index in range(2, 9):
        _touch_capture(tmp_path, "2026.06.01", "PRODUCT_A", f"G{index:03d}")

    records = scan_ok_tree(tmp_path, start_date="2026-05-01")
    splits = split_by_group(records, seed=17, ratios=(0.6, 0.2, 0.2))

    locations: dict[str, set[str]] = {}
    for split_name, split_records in splits.items():
        for record in split_records:
            locations.setdefault(record.group_id, set()).add(split_name)

    assert all(len(split_names) == 1 for split_names in locations.values())
    assert sum(len(part) for part in splits.values()) == len(records)
    assert set(splits) == {"train", "calibration", "holdout"}
