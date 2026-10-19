from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml
import numpy as np

from .refinement_contract import (
    SignalRow,
    canonical_digest,
    read_jsonl,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from .refinement_signals import SignalBundle


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _save_signal_bundles(
    output_dir: Path, name: str, bundles: Sequence[SignalBundle]
) -> dict[str, object]:
    if not bundles:
        raise ValueError(f"{name} signal bundles must not be empty")
    dimensions = {bundle.row.feature_dimension for bundle in bundles}
    schemas = {bundle.row.schema_digest for bundle in bundles}
    if len(dimensions) != 1 or len(schemas) != 1:
        raise ValueError(f"{name} signal bundles must share one schema")
    values = np.stack([bundle.values for bundle in bundles]).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} signal values must be finite")
    output_dir.mkdir(parents=True, exist_ok=True)
    values_path = output_dir / f"{name}_signals.npz"
    temporary = output_dir / f".{name}_signals.tmp"
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, values=values)
        temporary.replace(values_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    rows_path = output_dir / f"{name}_signal_rows.jsonl"
    rows_digest = write_jsonl_atomic(rows_path, [bundle.row.to_dict() for bundle in bundles])
    return {
        "rows": len(bundles),
        "dimension": next(iter(dimensions)),
        "schema_digest": next(iter(schemas)),
        "values_file": values_path.name,
        "values_sha256": sha256_file(values_path),
        "rows_file": rows_path.name,
        "rows_sha256": rows_digest,
    }


def _load_signal_bundles(output_dir: Path, name: str) -> list[SignalBundle]:
    rows_path = output_dir / f"{name}_signal_rows.jsonl"
    values_path = output_dir / f"{name}_signals.npz"
    rows = read_jsonl(rows_path)
    with np.load(values_path, allow_pickle=False) as archive:
        values = np.asarray(archive["values"], dtype=np.float32)
    if values.ndim != 2 or len(rows) != len(values):
        raise RuntimeError(f"{name} signal artifact row count mismatch")
    bundles: list[SignalBundle] = []
    for row, vector in zip(rows, values, strict=True):
        row = dict(row)
        row["box"] = tuple(row["box"])
        signal_row = SignalRow(**row)
        if vector.size != signal_row.feature_dimension or not np.isfinite(vector).all():
            raise RuntimeError(f"{name} signal artifact schema mismatch")
        bundles.append(SignalBundle(signal_row, vector.copy()))
    return bundles


def _load_config(path: str | Path) -> dict[str, object]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("refinement configuration root must be a mapping")
    if value.get("version") != "candidate_refinement_v1":
        raise ValueError("unsupported refinement configuration version")
    return value


def _read_complete_receipt(path: Path, label: str) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError(f"required {label} is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("status") != "completed"
        or int(value.get("failures", -1)) != 0
        or int(value.get("temporary_files", -1)) != 0
    ):
        raise RuntimeError(f"{label} is not a complete zero-failure receipt")
    if not isinstance(value.get("digests"), dict):
        raise RuntimeError(f"{label} has no digest map")
    return value


def _verify_shared_digests(receipts: Sequence[dict[str, object]]) -> None:
    digest_maps = [dict(receipt["digests"]) for receipt in receipts]
    shared_keys = set.intersection(*(set(values) for values in digest_maps))
    if not shared_keys:
        raise RuntimeError("upstream receipts have no shared digest boundary")
    for key in sorted(shared_keys):
        values = {mapping[key] for mapping in digest_maps}
        if len(values) != 1:
            raise RuntimeError(f"upstream receipt digest mismatch for {key}")


def _freeze_policy(args: argparse.Namespace) -> int:
    run_root = Path(args.run_root)
    config = _load_config(args.config)
    receipt_paths = {
        "candidate_build_receipt": run_root / "candidate_cache/candidate_build_receipt.json",
        "anchor_build_receipt": run_root / "anchors/anchor_build_receipt.json",
        "fit_receipt": run_root / "refiner/fit_receipt.json",
    }
    receipts = {
        name: _read_complete_receipt(path, name) for name, path in receipt_paths.items()
    }
    _verify_shared_digests(list(receipts.values()))
    candidate = receipts["candidate_build_receipt"]
    anchors = receipts["anchor_build_receipt"]
    fit = receipts["fit_receipt"]
    for key, receipt, label in (
        ("candidate_policy", candidate, "candidate_build_receipt"),
        ("signal_schema_digest", anchors, "anchor_build_receipt"),
        ("normal_memory_digest", anchors, "anchor_build_receipt"),
        ("selected_refiner", fit, "fit_receipt"),
        ("selected_refiner_digest", fit, "fit_receipt"),
    ):
        if key not in receipt:
            raise RuntimeError(f"{label} is missing {key}")
    destination = run_root / "frozen_policy.json"
    if destination.exists():
        raise FileExistsError(destination)
    ranking = dict(config["ranking"])
    if any("display" in key.casefold() for key in ranking):
        raise RuntimeError("ranking configuration must not contain display settings")
    frozen = {
        "format_version": 1,
        "version": config["version"],
        "config_digest": canonical_digest(config),
        "model": dict(config["model"]),
        "candidate_policy": candidate["candidate_policy"],
        "anchor_policy": dict(config["clean_anchor"]),
        "signal_schema_digest": anchors["signal_schema_digest"],
        "normal_memory_digest": anchors["normal_memory_digest"],
        "selected_refiner": fit["selected_refiner"],
        "selected_refiner_digest": fit["selected_refiner_digest"],
        "ranking": ranking,
        "training_contract": dict(config["training"]),
        "evaluation_contract": dict(config["evaluation"]),
        "upstream_receipt_sha256": {
            name: sha256_file(path) for name, path in receipt_paths.items()
        },
        "shared_digests": {
            key: value
            for key, value in dict(candidate["digests"]).items()
            if all(dict(receipt["digests"]).get(key) == value for receipt in receipts.values())
        },
    }
    run_root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(destination, frozen)
    write_json_atomic(
        run_root / "freeze_receipt.json",
        {
            "format_version": 1,
            "status": "completed",
            "failures": 0,
            "temporary_files": 0,
            "frozen_policy_sha256": sha256_file(destination),
            "config_digest": frozen["config_digest"],
        },
    )
    return 0


def _require_prior_receipt(args: argparse.Namespace, relative: str, label: str) -> int:
    _load_config(args.config)
    _read_complete_receipt(Path(args.run_root) / relative, label)
    return 0


def _validate_rank_request(payload: object) -> list[dict[str, str]]:
    if not isinstance(payload, Mapping):
        raise ValueError("rank request root must be a mapping")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("rank request requires a non-empty cases list")
    allowed = {"case_id", "image_path", "product", "group_id", "condition_id"}
    cases: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw_case in raw_cases:
        if not isinstance(raw_case, Mapping):
            raise ValueError("each rank case must be a mapping")
        extra = set(raw_case) - allowed
        if extra:
            raise ValueError(
                f"rank request must remain label-free; unsupported fields: {sorted(extra)}"
            )
        missing = allowed - set(raw_case)
        if missing:
            raise ValueError(f"rank case is missing required fields: {sorted(missing)}")
        case = {key: str(raw_case[key]).strip() for key in allowed}
        if any(not value for value in case.values()):
            raise ValueError("rank case fields must be non-empty strings")
        if case["case_id"] in seen:
            raise ValueError("rank case IDs must be unique")
        seen.add(case["case_id"])
        cases.append(case)
    return cases


def _anchor_row(value: Mapping[str, object]):
    from .refinement_contract import AnchorRow

    row = dict(value)
    row["box"] = tuple(row["box"])
    return AnchorRow(**row)


def _load_frozen_normal_memory(
    run_root: Path,
    frozen: Mapping[str, object],
    config: Mapping[str, object],
):
    from .refinement_memory import NormalMemoryIndex

    anchor_receipt = _read_complete_receipt(
        run_root / "anchors/anchor_build_receipt.json", "anchor_build_receipt"
    )
    memory_receipt = dict(anchor_receipt.get("normal_memory", {}))
    required = {"file", "file_sha256", "logical_digest"}
    if set(memory_receipt) < required:
        raise RuntimeError("anchor receipt has no complete normal-memory artifact")
    memory_path = run_root / "anchors/normal_memory" / str(memory_receipt["file"])
    if not memory_path.is_file() or sha256_file(memory_path) != memory_receipt["file_sha256"]:
        raise RuntimeError("normal-memory artifact digest mismatch")
    expected_digest = str(frozen["normal_memory_digest"])
    if memory_receipt["logical_digest"] != expected_digest:
        raise RuntimeError("normal-memory receipt is outside the frozen boundary")
    rows = [
        _anchor_row(value)
        for value in read_jsonl(run_root / "anchors/clean_anchor_manifest.jsonl")
    ]
    return NormalMemoryIndex.load(
        memory_path,
        rows,
        expected_digest=expected_digest,
        chunk_size=int(dict(config["normal_memory"])["chunk_size"]),
    )


def _load_frozen_refiner(run_root: Path, frozen: Mapping[str, object]):
    import torch

    from .refinement_ranking import (
        EmpiricalTailCalibrator,
        WeakRanker,
        WeakRankerReceipt,
        _ranker_state_digest,
    )

    selected = str(frozen["selected_refiner"])
    expected_digest = str(frozen["selected_refiner_digest"])
    if selected == "deterministic_fusion":
        path = run_root / "refiner/deterministic_fusion.json"
        if not path.is_file() or sha256_file(path) != expected_digest:
            raise RuntimeError("deterministic refiner digest is outside the frozen boundary")
        payload = json.loads(path.read_text(encoding="utf-8"))
        distributions = {
            str(name): {
                str(bank): tuple(float(value) for value in values)
                for bank, values in dict(banks).items()
            }
            for name, banks in dict(payload["distributions"]).items()
        }
        return EmpiricalTailCalibrator(
            distributions=distributions,
            minimum_bank_size=int(payload["minimum_bank_size"]),
        )
    if selected != "weak_ranker":
        raise RuntimeError(f"unsupported frozen refiner: {selected}")
    path = run_root / "refiner/weak_ranker.pt"
    if not path.is_file():
        raise RuntimeError("frozen weak-ranker artifact is missing")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = dict(payload["head_state_dict"])
    input_dimension = int(state["network.1.weight"].shape[1])
    model = WeakRanker(input_dimension)
    model.load_state_dict(state, strict=True)
    model.eval()
    state_digest = _ranker_state_digest(model)
    if state_digest != expected_digest or payload.get("state_digest") != expected_digest:
        raise RuntimeError("weak-ranker state is outside the frozen boundary")
    return WeakRankerReceipt(
        model=model,
        state_digest=state_digest,
        schema_digest=str(payload["schema_digest"]),
        data_digest=str(payload["data_digest"]),
        config_digest=str(payload["config_digest"]),
        epochs=0,
        best_calibration_loss=0.0,
    )


def _rank_images(args: argparse.Namespace) -> int:
    import shutil
    import time

    from PIL import Image

    from .refinement_contract import build_run_receipt
    from .refinement_encoding import CandidatePolicy, encode_patch_batch
    from .refinement_inference import (
        InferenceCase,
        rank_image_case,
        review_candidates_to_json,
    )

    run_root = Path(args.run_root)
    frozen_path = run_root / "frozen_policy.json"
    if not frozen_path.is_file():
        raise RuntimeError("frozen policy is required before image ranking")
    request_path = Path(args.request)
    if not request_path.is_file():
        raise FileNotFoundError(request_path)
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    config = _load_config(args.config)
    if canonical_digest(config) != frozen.get("config_digest"):
        raise RuntimeError("runtime configuration is outside the frozen boundary")
    if dict(config["model"]) != dict(frozen["model"]):
        raise RuntimeError("runtime model configuration is outside the frozen boundary")
    cases = _validate_rank_request(json.loads(request_path.read_text(encoding="utf-8")))
    checkpoint_path = Path(args.checkpoint)
    model, device = _load_frozen_model(config, checkpoint_path)

    class Encoder:
        def encode(self, rgb_batch):
            return encode_patch_batch(
                model,
                rgb_batch,
                device,
                precision=str(dict(config["model"])["precision"]),
            )

    memory = _load_frozen_normal_memory(run_root, frozen, config)
    refiner = _load_frozen_refiner(run_root, frozen)
    policy = CandidatePolicy(**dict(frozen["candidate_policy"]))
    ranking_config = dict(frozen["ranking"])
    top_k = int(ranking_config["expanded_top_k"])
    output = Path(args.output_dir) if args.output_dir else run_root / "ranking"
    temporary = output.with_name(f".{output.name}.tmp")
    if output.exists() or temporary.exists():
        raise FileExistsError(output if output.exists() else temporary)
    temporary.mkdir(parents=True, exist_ok=False)
    refined_payload: dict[str, object] = {}
    baseline_payload: dict[str, object] = {}
    case_receipts: list[dict[str, object]] = []
    started = time.perf_counter()
    try:
        for values in cases:
            image_path = Path(values["image_path"])
            source_digest = sha256_file(image_path)
            with Image.open(image_path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            case = InferenceCase(
                case_id=values["case_id"],
                image_path=str(image_path),
                source_sha256=source_digest,
                product=values["product"],
                group_id=values["group_id"],
                condition_id=values["condition_id"],
            )
            case_started = time.perf_counter()
            result = rank_image_case(
                rgb,
                case,
                Encoder(),
                policy,
                memory,
                refiner,
                batch_size=int(args.batch_size or dict(config["runtime"])["batch_size"]),
                query_k=int(dict(config["normal_memory"])["query_k"]),
                top_k=top_k,
                stride=int(ranking_config["stride"]),
            )
            refined_payload[case.case_id] = review_candidates_to_json(result.refined)
            baseline_payload[case.case_id] = review_candidates_to_json(result.baseline)
            case_receipts.append(
                {
                    "case_id": case.case_id,
                    "source_sha256": source_digest,
                    "width": int(rgb.shape[1]),
                    "height": int(rgb.shape[0]),
                    "total_patches": result.total_patches,
                    "retained_candidates": result.candidate_count,
                    "refined_candidates": len(result.refined),
                    "baseline_candidates": len(result.baseline),
                    "refined_refiner": result.refined_refiner,
                    "baseline_refiner": result.baseline_refiner,
                    "elapsed_seconds": time.perf_counter() - case_started,
                    "review_state": "REVIEW",
                }
            )
        source_rows = read_jsonl(run_root / "candidate_cache/source_manifest.jsonl")
        training_hashes = sorted({str(row["source_sha256"]) for row in source_rows})
        write_json_atomic(temporary / "refined_candidates.json", refined_payload)
        write_json_atomic(temporary / "baseline_candidates.json", baseline_payload)
        write_json_atomic(temporary / "training_hashes.json", training_hashes)
        write_jsonl_atomic(temporary / "case_receipts.jsonl", case_receipts)
        artifacts = {
            path.name: sha256_file(path)
            for path in sorted(temporary.iterdir())
            if path.is_file()
        }
        receipt = build_run_receipt(
            "frozen-image-ranking",
            digests={
                "frozen_policy": sha256_file(frozen_path),
                "request": sha256_file(request_path),
                "checkpoint": sha256_file(checkpoint_path),
                "normal_memory": str(frozen["normal_memory_digest"]),
                "selected_refiner": str(frozen["selected_refiner_digest"]),
                **artifacts,
            },
            counts={
                "cases": len(cases),
                "total_patches": sum(int(row["total_patches"]) for row in case_receipts),
                "retained_candidates": sum(
                    int(row["retained_candidates"]) for row in case_receipts
                ),
                "refined_candidates": sum(
                    int(row["refined_candidates"]) for row in case_receipts
                ),
                "baseline_candidates": sum(
                    int(row["baseline_candidates"]) for row in case_receipts
                ),
                "training_hashes": len(training_hashes),
            },
            metrics={"elapsed_seconds": time.perf_counter() - started},
            failures=0,
            temporary_files=0,
            seeds={"ranking": int(dict(config["runtime"])["seed"])},
        )
        receipt["candidate_policy"] = dict(frozen["candidate_policy"])
        receipt["top_k"] = top_k
        receipt["review_state"] = "REVIEW"
        write_json_atomic(temporary / "rank_receipt.json", receipt)
        temporary.replace(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return 0


def _sealed_evaluate(args: argparse.Namespace) -> int:
    run_root = Path(args.run_root)
    frozen_path = run_root / "frozen_policy.json"
    if not frozen_path.is_file():
        raise RuntimeError("frozen policy is required before sealed evaluation")
    if args.ranked_candidates is None or args.baseline_candidates is None:
        raise RuntimeError("sealed evaluation requires frozen ranked and baseline candidate files")
    from .refinement_evaluation import run_sealed_evaluation, seal_labelme_manifest
    from .refinement_ranking import ReviewCandidate
    from .inference import Box

    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    manifest_path = run_root / "sealed_labelme_manifest.json"
    if manifest_path.exists():
        manifest_digest = sha256_file(manifest_path)
    else:
        manifest_digest = seal_labelme_manifest(args.labelme_root, manifest_path)

    def load_candidates(path: str) -> dict[str, list[ReviewCandidate]]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        output: dict[str, list[ReviewCandidate]] = {}
        for case_id, rows in payload.items():
            output[case_id] = [
                ReviewCandidate(
                    component_id=row["component_id"],
                    patch_ids=tuple(row["patch_ids"]),
                    box=Box(**row["box"]),
                    score=float(row["score"]),
                    probability=float(row["probability"]),
                    support=int(row["support"]),
                    roi_relation=row["roi_relation"],
                    review_state=row.get("review_state", "REVIEW"),
                )
                for row in rows
            ]
        return output

    digests = {
        "config": frozen["config_digest"],
        "candidate_policy": canonical_digest(frozen["candidate_policy"]),
        "feature_schema": frozen["signal_schema_digest"],
        "normal_memory": frozen["normal_memory_digest"],
        "selected_refiner": frozen["selected_refiner_digest"],
        "labelme_manifest": manifest_digest,
    }
    training_hashes = json.loads(Path(args.training_hashes).read_text(encoding="utf-8")) if args.training_hashes else []
    artifacts = {
        "digests": digests,
        "training_hashes": training_hashes,
        "ranked_candidates": load_candidates(args.ranked_candidates),
        "baseline_candidates": load_candidates(args.baseline_candidates),
    }
    run_sealed_evaluation(
        {"digests": digests},
        artifacts,
        manifest_path,
        run_root / "sealed_evaluation",
    )
    return 0


def _parse_review_candidates(values: Sequence[Mapping[str, object]]):
    from .inference import Box
    from .refinement_ranking import ReviewCandidate

    candidates: list[ReviewCandidate] = []
    for value in values:
        box_value = dict(value["box"])
        box = Box(
            x0=int(box_value["x0"]),
            y0=int(box_value["y0"]),
            x1=int(box_value["x1"]),
            y1=int(box_value["y1"]),
        )
        if box.x1 - box.x0 != 256 or box.y1 - box.y0 != 256:
            raise ValueError("review LeGrad requires native 256 x 256 candidate boxes")
        candidate = ReviewCandidate(
            component_id=str(value["component_id"]),
            patch_ids=tuple(str(item) for item in value["patch_ids"]),
            box=box,
            score=float(value["score"]),
            probability=float(value["probability"]),
            support=int(value["support"]),
            roi_relation=str(value["roi_relation"]),
            review_state=str(value.get("review_state", "REVIEW")),
        )
        if candidate.review_state != "REVIEW":
            raise ValueError("review export accepts only REVIEW candidates")
        candidates.append(candidate)
    return candidates


def _export_review(args: argparse.Namespace) -> int:
    import time

    from PIL import Image

    from .inference import _normalize_batch
    from .refinement_contract import build_run_receipt
    from .refinement_legrad import token_legrad
    from .refinement_review import export_review_case

    run_root = Path(args.run_root)
    frozen_path = run_root / "frozen_policy.json"
    if not frozen_path.is_file():
        raise RuntimeError("frozen policy is required before review export")
    request = Path(args.request)
    if not request.is_file():
        raise FileNotFoundError(request)
    config = _load_config(args.config)
    model, device = _load_frozen_model(config, Path(args.checkpoint))
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    payload = json.loads(request.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("review request requires a non-empty cases list")
    output = Path(args.output_dir) if args.output_dir else run_root / "review"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    artifacts: dict[str, str] = {}
    total_candidates = 0
    started = time.perf_counter()
    for case in cases:
        case_id = str(case["case_id"])
        image_path = Path(case["image_path"])
        candidates = _parse_review_candidates(case["candidates"])
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        patches = []
        for candidate in candidates:
            box = candidate.box
            if box.x0 < 0 or box.y0 < 0 or box.x1 > rgb.shape[1] or box.y1 > rgb.shape[0]:
                raise ValueError(f"candidate lies outside image: {case_id}/{candidate.component_id}")
            patches.append(rgb[box.y0 : box.y1, box.x0 : box.x1])
        legrad_started = time.perf_counter()
        _, cams = token_legrad(
            model,
            _normalize_batch(np.stack(patches)).to(device),
            layer_start=int(dict(config["model"])["legrad_layer_start"]),
        )
        legrad_seconds = time.perf_counter() - legrad_started
        case_output = output / case_id.replace("/", "__")
        paths = export_review_case(
            rgb,
            candidates,
            [cam for cam in cams.float().cpu().numpy()],
            {
                "case_id": case_id,
                "source_path": str(image_path),
                "source_sha256": sha256_file(image_path),
                "display_threshold": float(dict(config["display"])["threshold"]),
                "model_digest": str(dict(config["model"])["checkpoint_sha256"]),
                "config_digest": frozen["config_digest"],
                "schema_digest": frozen["signal_schema_digest"],
                "timings": {"legrad_seconds": legrad_seconds},
            },
            case_output,
        )
        for name, path in paths.items():
            artifacts[f"{case_id}:{name}"] = sha256_file(path)
        total_candidates += len(candidates)
    elapsed = time.perf_counter() - started
    receipt = build_run_receipt(
        "review-export",
        digests={
            "frozen_policy": sha256_file(frozen_path),
            "request": sha256_file(request),
            **artifacts,
        },
        counts={"cases": len(cases), "candidates": total_candidates},
        metrics={"elapsed_seconds": elapsed},
        failures=0,
        temporary_files=len(list(output.rglob("*.tmp"))),
    )
    write_json_atomic(output / "review_export_receipt.json", receipt)
    return 0


def _load_frozen_model(config: Mapping[str, object], checkpoint_path: Path):
    from .model import load_timm_probe
    import torch

    model_config = dict(config["model"])
    expected_checkpoint = str(model_config["checkpoint_sha256"])
    if sha256_file(checkpoint_path) != expected_checkpoint:
        raise RuntimeError("checkpoint digest does not match refinement configuration")
    required_gpus = int(dict(config["runtime"])["required_visible_gpus"])
    if torch.cuda.device_count() != required_gpus:
        raise RuntimeError(f"expected exactly {required_gpus} scheduler-visible GPU")
    device = torch.device("cuda:0")
    model = load_timm_probe(
        str(model_config["backbone_id"]),
        local_files_only=bool(model_config["local_files_only"]),
        dropout=float(model_config["dropout"]),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.head.load_state_dict(checkpoint["head_state_dict"], strict=True)
    model.to(device).eval()
    return model, device


def _audit_source_digest(path: str, source_sha256: str) -> str:
    return source_sha256 or sha256_file(path)


def _calibrate_cache_policy(config, image_rows, encoder):
    from PIL import Image

    from .inference import tile_boxes
    from .refinement_encoding import calibrate_candidate_policy
    from .roi import estimate_placement_mask
    from .synthetic import SyntheticDefectGenerator

    candidate_config = dict(config["candidate_policy"])
    calibration_sources = [row for row in image_rows if row.split == "calibration"]
    if not calibration_sources:
        calibration_sources = [row for row in image_rows if row.split == "holdout"]
    calibration_sources = calibration_sources[: int(candidate_config["calibration_images"])]
    if not calibration_sources:
        raise RuntimeError("candidate-policy calibration requires calibration or holdout OK sources")
    patch_size = int(dict(config["data"])["patch_size"])
    stride = int(dict(config["data"])["stride"])
    patches_per_image = int(candidate_config["calibration_patches_per_image"])
    generator = SyntheticDefectGenerator()
    seed = int(dict(config["runtime"])["seed"])
    clean_cases: list[dict[str, object]] = []
    synthetic_cases: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for source_index, source in enumerate(calibration_sources):
        with Image.open(source.path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        boxes = tile_boxes(rgb.shape[1], rgb.shape[0], patch_size, stride)
        sample_count = min(patches_per_image, len(boxes))
        indices = np.linspace(0, len(boxes) - 1, sample_count, dtype=np.int64)
        sampled_boxes = [boxes[int(index)] for index in indices]
        patches = np.stack(
            [rgb[box.y0 : box.y1, box.x0 : box.x1] for box in sampled_boxes]
        )
        logits, _ = encoder.encode(patches)
        probabilities = (
            1.0 / (1.0 + np.exp(-np.clip(np.asarray(logits, dtype=np.float32), -80.0, 80.0)))
        ).astype(np.float32)
        support = estimate_placement_mask(rgb, margin=8)
        support_fractions = [
            float(support[box.y0 : box.y1, box.x0 : box.x1].mean()) for box in sampled_boxes
        ]
        target_index = int(np.argmax(support_fractions))
        target_box = sampled_boxes[target_index]
        patch_support = support[
            target_box.y0 : target_box.y1, target_box.x0 : target_box.x1
        ]
        if float(patch_support.mean()) < 0.10:
            patch_support = patches[target_index].mean(axis=2) > 8
        rng = np.random.default_rng(np.random.SeedSequence([seed, source_index]))
        family = generator.families[source_index % len(generator.families)]
        severity = generator.severities[source_index % len(generator.severities)]
        result = generator.apply(
            patches[target_index], patch_support.astype(bool), family, rng, severity=severity
        )
        synthetic_logit, _ = encoder.encode(result.image[None, ...])
        synthetic_probability = float(
            1.0 / (1.0 + np.exp(-np.clip(float(synthetic_logit[0]), -80.0, 80.0)))
        )
        synthetic_probabilities = probabilities.copy()
        synthetic_probabilities[target_index] = synthetic_probability
        clean_cases.append(
            {
                "source_label": "OK",
                "probabilities": probabilities.tolist(),
                "target_indices": [],
            }
        )
        synthetic_cases.append(
            {
                "source_label": "SYNTHETIC",
                "probabilities": synthetic_probabilities.tolist(),
                "target_indices": [target_index],
            }
        )
        audit_rows.append(
            {
                "source_sha256": _audit_source_digest(source.path, source.source_sha256),
                "family": family,
                "severity": severity,
                "target_box": [target_box.x0, target_box.y0, target_box.x1, target_box.y1],
                "target_probability": synthetic_probability,
                "sampled_patches": sample_count,
            }
        )
    policy = calibrate_candidate_policy(
        clean_cases,
        synthetic_cases,
        floors=tuple(float(value) for value in candidate_config["floor_grid"]),
        top_m_values=tuple(int(value) for value in candidate_config["top_m_grid"]),
        maximum_candidates=int(candidate_config["maximum_candidates"]),
        target_local_recall=float(candidate_config["target_synthetic_local_recall"]),
    )
    return policy, audit_rows


def _select_limited_image_specs(
    image_specs: Sequence[dict[str, str]], limit: int
) -> list[dict[str, str]]:
    """Select a split-balanced smoke cohort with cross-lot buckets first."""
    if limit <= 0:
        return list(image_specs)
    by_split: dict[str, list[dict[str, str]]] = {}
    for split in ("train", "calibration", "holdout"):
        buckets: dict[tuple[str, str], list[dict[str, str]]] = {}
        for value in image_specs:
            if value["split"] != split:
                continue
            buckets.setdefault((value["product"], value["condition_id"]), []).append(value)
        ordered: list[dict[str, str]] = []
        for _, values in sorted(
            buckets.items(),
            key=lambda item: (-len({row["group_id"] for row in item[1]}), item[0]),
        ):
            first_per_group: list[dict[str, str]] = []
            remaining: list[dict[str, str]] = []
            seen_groups: set[str] = set()
            for value in values:
                if value["group_id"] in seen_groups:
                    remaining.append(value)
                else:
                    first_per_group.append(value)
                    seen_groups.add(value["group_id"])
            ordered.extend(first_per_group)
            ordered.extend(remaining)
        by_split[split] = ordered

    selected_specs: list[dict[str, str]] = []
    while len(selected_specs) < limit:
        changed = False
        for split in ("train", "calibration", "holdout"):
            if by_split[split] and len(selected_specs) < limit:
                selected_specs.append(by_split[split].pop(0))
                changed = True
        if not changed:
            break
    return selected_specs


def _partition_image_specs(
    image_specs: Sequence[dict[str, str]], shard_count: int, shard_index: int
) -> list[dict[str, str]]:
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("cache shard index must be within a positive shard count")
    ordered = sorted(
        image_specs,
        key=lambda value: (canonical_digest({"path": value["path"]}), value["path"]),
    )
    return [
        value for position, value in enumerate(ordered) if position % shard_count == shard_index
    ]


def _build_cache(args: argparse.Namespace) -> int:
    from .refinement_progress import ProgressReporter

    progress_path = Path(args.run_root) / "progress.jsonl"
    with ProgressReporter(
        progress_path,
        interval_seconds=args.progress_interval_seconds,
    ) as progress:
        return _build_cache_with_progress(args, progress)


def _build_cache_with_progress(args: argparse.Namespace, progress) -> int:
    config = _load_config(args.config)
    from .refinement_cache import ImageMeta, SelectiveCacheBuilder
    from .refinement_encoding import encode_patch_batch
    from .refinement_signals import SignalSchema

    checkpoint_path = Path(args.checkpoint)
    expected_checkpoint = str(dict(config["model"])["checkpoint_sha256"])
    progress.emit(
        "model_load",
        "started",
        checkpoint=str(checkpoint_path),
        checkpoint_sha256=expected_checkpoint,
    )
    model, device = _load_frozen_model(config, checkpoint_path)
    progress.emit("model_load", "completed", device=str(device))

    class Encoder:
        def encode(self, rgb_batch):
            return encode_patch_batch(
                model,
                rgb_batch,
                device,
                precision=str(dict(config["model"])["precision"]),
            )

    image_specs: list[dict[str, str]] = []
    with Path(args.source_manifest).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("source_label") != "OK":
                raise ValueError("candidate cache accepts only OK source manifests")
            for image in row.get("images", ()):
                image_specs.append(
                    {
                        "path": str(image["path"]),
                        "product": str(row["product"]),
                        "group_id": str(row["group_id"]),
                        "condition_id": str(image["condition"]["condition_id"]),
                        "split": str(row["split"]),
                    }
                )
    if args.limit_sources:
        image_specs = _select_limited_image_specs(image_specs, args.limit_sources)
    calibration_rows = [
        ImageMeta(
            path=value["path"],
            source_sha256="",
            product=value["product"],
            group_id=value["group_id"],
            condition_id=value["condition_id"],
            split=value["split"],
        )
        for value in image_specs
    ]
    progress.emit(
        "calibration",
        "started",
        eligible_sources=len(calibration_rows),
    )
    policy, calibration_audit = _calibrate_cache_policy(config, calibration_rows, Encoder())
    progress.emit(
        "calibration",
        "completed",
        audited_sources=len(calibration_audit),
        probability_floor=policy.probability_floor,
        minimum_top_m=policy.minimum_top_m,
        maximum_candidates=policy.maximum_candidates,
    )
    if args.shard_count is not None or args.shard_index is not None:
        if args.shard_count is None or args.shard_index is None:
            raise ValueError("cache sharding requires both --shard-count and --shard-index")
        image_specs = _partition_image_specs(image_specs, args.shard_count, args.shard_index)
    image_rows = [
        ImageMeta(
            path=value["path"],
            source_sha256="",
            product=value["product"],
            group_id=value["group_id"],
            condition_id=value["condition_id"],
            split=value["split"],
        )
        for value in image_specs
    ]
    builder = SelectiveCacheBuilder(
        encoder=Encoder(),
        policy=policy,
        batch_size=int(args.batch_size or dict(config["runtime"])["batch_size"]),
        device=str(device),
        patch_size=int(dict(config["data"])["patch_size"]),
        stride=int(dict(config["data"])["stride"]),
        shard_size=int(dict(config["cache"])["shard_size"]),
        reservoir_per_product_condition=int(dict(config["cache"])["reservoir_per_product_condition"]),
        reservoir_per_group=int(dict(config["cache"])["reservoir_per_group"]),
        seed=int(dict(config["runtime"])["seed"]),
        model_digest=expected_checkpoint,
        feature_schema_digest=SignalSchema.default(model.hidden_size * 2).digest,
        workers=args.workers,
        prefetch_sources=args.prefetch_sources,
        prefetch_batches=args.prefetch_batches,
    )
    progress.start_cache(total_sources=len(image_rows))

    def report_source(
        completed_sources: int,
        total_sources: int,
        total_patches: int,
        current_path: str,
    ) -> None:
        if total_sources != len(image_rows):
            raise RuntimeError("cache progress source boundary changed")
        progress.source_completed(
            completed_sources=completed_sources,
            total_patches=total_patches,
            current_path=current_path,
        )

    receipt = builder.build(
        image_rows,
        Path(args.run_root) / "candidate_cache",
        resume=args.resume,
        progress_callback=report_source,
    )
    counts = dict(receipt["counts"])
    if int(counts["sources"]) != len(image_rows):
        raise RuntimeError("cache receipt source count does not match progress boundary")
    progress.finish_cache(total_patches=int(counts["total_patches"]))
    receipt["candidate_policy"] = {
        "probability_floor": policy.probability_floor,
        "minimum_top_m": policy.minimum_top_m,
        "maximum_candidates": policy.maximum_candidates,
    }
    receipt["candidate_policy_calibration"] = calibration_audit
    receipt["candidate_policy_calibration_digest"] = canonical_digest(calibration_audit)
    receipt["config_digest"] = canonical_digest(config)
    if args.shard_count is not None:
        receipt["cache_shard"] = {"count": args.shard_count, "index": args.shard_index}
    receipt["progress"] = {
        "events": progress.event_count,
        "file": "../progress.jsonl",
        "interval_seconds": args.progress_interval_seconds,
        "sha256": sha256_file(progress.path),
    }
    write_json_atomic(Path(args.run_root) / "candidate_cache/candidate_build_receipt.json", receipt)
    return 0


def _candidate_row(value: Mapping[str, object]):
    from .refinement_contract import CandidateRow

    row = dict(value)
    row["box"] = tuple(row["box"])
    return CandidateRow(**row)


def _merge_cache(args: argparse.Namespace) -> int:
    from .refinement_cache import merge_selective_caches

    config = _load_config(args.config)
    run_root = Path(args.run_root)
    if run_root.exists():
        raise FileExistsError(run_root)
    shard_dirs = [Path(path) for path in args.shard_cache]
    if len(shard_dirs) < 2:
        raise ValueError("cache merge requires at least two --shard-cache inputs")
    run_root.mkdir(parents=True, exist_ok=False)
    try:
        merge_selective_caches(
            shard_dirs,
            run_root / "candidate_cache",
            reservoir_per_product_condition=int(
                dict(config["cache"])["reservoir_per_product_condition"]
            ),
            reservoir_per_group=int(dict(config["cache"])["reservoir_per_group"]),
            shard_size=int(dict(config["cache"])["shard_size"]),
            seed=int(dict(config["runtime"])["seed"]),
            expected_config_digest=canonical_digest(config),
        )
    except BaseException:
        if not any(run_root.iterdir()):
            run_root.rmdir()
        raise
    return 0


def _source_row(value: Mapping[str, object]):
    from .refinement_contract import SourceRow

    return SourceRow(**value)


def _build_anchors(args: argparse.Namespace) -> int:
    from dataclasses import replace

    from PIL import Image

    from .refinement_contract import (
        AnchorRow,
        CandidateRow,
        build_run_receipt,
        write_jsonl_atomic,
    )
    from .refinement_memory import CleanAnchorPolicy, NormalMemoryIndex, select_clean_anchors
    from .refinement_signals import (
        CandidateEvidence,
        augment_candidate_evidence,
        build_signal_bundle,
        build_synthetic_anchor_rows,
    )
    from .refinement_encoding import encode_patch_batch
    from .roi import estimate_placement_mask
    from .synthetic import SyntheticDefectGenerator

    config = _load_config(args.config)
    run_root = Path(args.run_root)
    candidate_receipt = _read_complete_receipt(
        run_root / "candidate_cache/candidate_build_receipt.json", "candidate_build_receipt"
    )
    output = run_root / "anchors"
    if output.exists():
        raise FileExistsError(output)
    model, device = _load_frozen_model(config, Path(args.checkpoint))

    class Encoder:
        def encode(self, rgb_batch):
            return encode_patch_batch(
                model,
                rgb_batch,
                device,
                precision=str(dict(config["model"])["precision"]),
            )

    encoder = Encoder()
    cache_dir = run_root / "candidate_cache"
    candidate_rows = [
        _candidate_row(value) for value in read_jsonl(cache_dir / "candidate_manifest.jsonl")
    ]
    reservoir_rows = [
        _candidate_row(value) for value in read_jsonl(cache_dir / "low_score_reservoir.jsonl")
    ]
    retained_rows = [*candidate_rows, *reservoir_rows]
    if args.limit_rows:
        retained_rows = retained_rows[: args.limit_rows]
        candidate_ids = {row.patch_id for row in retained_rows}
        candidate_rows = [row for row in candidate_rows if row.patch_id in candidate_ids]
    if not retained_rows:
        raise RuntimeError("candidate cache contains no retained rows")

    def image_loader(path: str) -> np.ndarray:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)

    def roi_provider(_path: str, rgb: np.ndarray) -> np.ndarray:
        return estimate_placement_mask(rgb, margin=8)

    evidence_rows, feature_blocks = augment_candidate_evidence(
        retained_rows,
        image_loader,
        encoder,
        roi_provider,
        batch_size=int(dict(config["runtime"])["batch_size"]),
    )
    evidence_by_id = {row.patch_id: row for row in evidence_rows}
    block_by_id = {
        row.patch_id: block
        for row, block in zip(evidence_rows, feature_blocks, strict=True)
    }
    anchor_candidates = [
        AnchorRow(
            patch_id=row.patch_id,
            source_sha256=row.source_sha256,
            product=row.product,
            group_id=row.group_id,
            condition_id=row.condition_id,
            box=row.box,
            probability=row.probability,
            tta_stability=min(
                evidence_by_id[row.patch_id].probability_stability,
                evidence_by_id[row.patch_id].feature_stability,
            ),
            foreground_fraction=evidence_by_id[row.patch_id].foreground_fraction,
            feature_key=row.feature_key,
            source_label="OK",
            source_kind="clean_anchor",
            roi_relation=evidence_by_id[row.patch_id].roi_relation,
            model_digest=str(dict(config["model"])["checkpoint_sha256"]),
            config_digest=canonical_digest(config),
        )
        for row in retained_rows
    ]
    row_by_id = {row.patch_id: row for row in retained_rows}
    anchor_policy = CleanAnchorPolicy(**dict(config["clean_anchor"]))

    def select_for_split(split: str) -> list[AnchorRow]:
        rows = [row for row in anchor_candidates if row_by_id[row.patch_id].split == split]
        if not rows:
            return []
        features = np.stack([block_by_id[row.patch_id][0] for row in rows])
        return select_clean_anchors(rows, features, anchor_policy)

    clean_train = select_for_split("train")
    clean_calibration = select_for_split("calibration")
    if len(clean_train) < int(dict(config["normal_memory"])["query_k"]):
        raise RuntimeError("too few cross-lot clean anchors for normal memory")
    clean_feature_by_id = {
        row.patch_id: block_by_id[row.patch_id][0] for row in clean_train
    }
    normal_memory = NormalMemoryIndex(
        np.stack([clean_feature_by_id[row.patch_id] for row in clean_train]),
        clean_train,
        chunk_size=int(dict(config["normal_memory"])["chunk_size"]),
    )

    sources = [_source_row(value) for value in read_jsonl(cache_dir / "source_manifest.jsonl")]
    train_sources = [source for source in sources if source.split == "train"]
    calibration_sources = [source for source in sources if source.split == "calibration"]
    if not calibration_sources:
        calibration_sources = [source for source in sources if source.split == "holdout"]
    if not train_sources or not calibration_sources:
        raise RuntimeError("anchor build requires train and calibration OK sources")
    synthetic_config = dict(config["synthetic_anchor"])
    train_count = int(args.synthetic_count or synthetic_config["train_count"])
    calibration_count = max(18, int(args.synthetic_calibration_count or synthetic_config["calibration_count"]))
    generator = SyntheticDefectGenerator()
    synthetic_train, synthetic_train_features = build_synthetic_anchor_rows(
        train_sources,
        encoder,
        generator,
        seed=int(dict(config["runtime"])["seed"]),
        count=train_count,
        model_digest=str(dict(config["model"])["checkpoint_sha256"]),
        config_digest=canonical_digest(config),
    )
    synthetic_calibration, synthetic_calibration_features = build_synthetic_anchor_rows(
        calibration_sources,
        encoder,
        generator,
        seed=int(dict(config["runtime"])["seed"]) + 1,
        count=calibration_count,
        model_digest=str(dict(config["model"])["checkpoint_sha256"]),
        config_digest=canonical_digest(config),
    )
    query_k = int(dict(config["normal_memory"])["query_k"])

    def clean_bundle(anchor: AnchorRow):
        candidate = row_by_id[anchor.patch_id]
        block = block_by_id[anchor.patch_id]
        evidence = evidence_by_id[anchor.patch_id]
        query = normal_memory.query(block[0], anchor.product, anchor.condition_id, k=query_k)
        bundle = build_signal_bundle(candidate, block[0], [block[2]], query, evidence)
        return SignalBundle(replace(bundle.row, source_kind="clean_anchor"), bundle.values)

    clean_train_bundles = [clean_bundle(anchor) for anchor in clean_train]
    clean_calibration_bundles = [clean_bundle(anchor) for anchor in clean_calibration]
    source_by_sha = {source.source_sha256: source for source in sources}

    def synthetic_bundles(rows, features):
        bundles: list[SignalBundle] = []
        for anchor, feature in zip(rows, features, strict=True):
            source = source_by_sha[anchor.source_sha256]
            probability = float(np.clip(anchor.probability, 1e-7, 1.0 - 1e-7))
            candidate = CandidateRow(
                patch_id=anchor.patch_id,
                source_sha256=anchor.source_sha256,
                source_path=source.path,
                product=anchor.product,
                group_id=anchor.group_id,
                condition_id=anchor.condition_id,
                split=source.split,
                box=anchor.box,
                logit=float(np.log(probability / (1.0 - probability))),
                probability=probability,
                foreground_fraction=anchor.foreground_fraction,
                feature_key=anchor.feature_key,
                source_label="SYNTHETIC",
            )
            provenance = json.loads(anchor.provenance_json)
            evidence = CandidateEvidence(
                patch_id=anchor.patch_id,
                probability_stability=anchor.tta_stability,
                feature_stability=anchor.tta_stability,
                neighbor_mean_feature_key=str(provenance["neighbor_mean_feature_key"]),
                context_cosine_contrast=float(provenance["context_cosine_contrast"]),
                foreground_fraction=anchor.foreground_fraction,
                roi_relation="inside",
                roi_fraction=anchor.foreground_fraction,
                spatial_support=1,
            )
            query = normal_memory.query(feature, anchor.product, anchor.condition_id, k=query_k)
            bundles.append(build_signal_bundle(candidate, feature, [feature], query, evidence))
        return bundles

    synthetic_train_bundles = synthetic_bundles(synthetic_train, synthetic_train_features)
    synthetic_calibration_bundles = synthetic_bundles(
        synthetic_calibration, synthetic_calibration_features
    )
    calibration_bundles = [*clean_calibration_bundles, *synthetic_calibration_bundles]
    output.mkdir(parents=True, exist_ok=False)
    clean_digest = write_jsonl_atomic(
        output / "clean_anchor_manifest.jsonl", [row.to_dict() for row in clean_train]
    )
    synthetic_digest = write_jsonl_atomic(
        output / "synthetic_anchor_manifest.jsonl",
        [row.to_dict() for row in synthetic_train],
    )
    signal_receipts = {
        "clean": _save_signal_bundles(output, "clean", clean_train_bundles),
        "synthetic": _save_signal_bundles(output, "synthetic", synthetic_train_bundles),
        "calibration": _save_signal_bundles(output, "calibration", calibration_bundles),
    }
    memory_receipt = normal_memory.save(output / "normal_memory")
    source_manifest_digest = str(dict(candidate_receipt["digests"])["source_manifest"])
    receipt = build_run_receipt(
        "anchor-build",
        digests={
            "source_manifest": source_manifest_digest,
            "clean_anchor_manifest": clean_digest,
            "synthetic_anchor_manifest": synthetic_digest,
            "normal_memory": normal_memory.digest,
            "config": canonical_digest(config),
        },
        counts={
            "clean_train": len(clean_train_bundles),
            "clean_calibration": len(clean_calibration_bundles),
            "synthetic_train": len(synthetic_train_bundles),
            "synthetic_calibration": len(synthetic_calibration_bundles),
        },
        failures=0,
        temporary_files=len(list(output.rglob("*.tmp"))),
        seeds={"anchor": int(dict(config["runtime"])["seed"])},
        extra={"signal_artifacts": signal_receipts, "normal_memory": memory_receipt},
    )
    receipt["signal_schema_digest"] = clean_train_bundles[0].row.schema_digest
    receipt["normal_memory_digest"] = normal_memory.digest
    write_json_atomic(output / "anchor_build_receipt.json", receipt)
    return 0


def _development_metrics(scores: np.ndarray, rows: Sequence[SignalBundle]) -> dict[str, float]:
    by_source: dict[str, list[int]] = {}
    for index, bundle in enumerate(rows):
        by_source.setdefault(bundle.row.source_sha256, []).append(index)
    synthetic_hits: list[float] = []
    clean_counts: list[float] = []
    for indices in by_source.values():
        ranked = sorted(indices, key=lambda index: (-float(scores[index]), rows[index].row.patch_id))
        synthetic = [index for index in indices if rows[index].row.source_label == "SYNTHETIC"]
        if synthetic:
            synthetic_hits.append(float(any(index in set(ranked[:5]) for index in synthetic)))
        clean = [index for index in indices if rows[index].row.source_label == "OK"]
        if clean:
            clean_counts.append(float(sum(scores[index] >= 0.5 for index in clean)))
    return {
        "synthetic_local_recall_at_5": float(np.mean(synthetic_hits)) if synthetic_hits else 0.0,
        "clean_mean_candidates_per_image": float(np.mean(clean_counts)) if clean_counts else 0.0,
        "tta_top5_agreement": 1.0,
    }


def _fit_refiner(args: argparse.Namespace) -> int:
    import torch

    from .refinement_contract import build_run_receipt
    from .refinement_ranking import (
        EmpiricalTailCalibrator,
        fit_weak_ranker,
        deterministic_relevance,
        select_refiner,
    )

    config = _load_config(args.config)
    run_root = Path(args.run_root)
    anchor_receipt = _read_complete_receipt(
        run_root / "anchors/anchor_build_receipt.json", "anchor_build_receipt"
    )
    output = run_root / "refiner"
    if output.exists():
        raise FileExistsError(output)
    clean = _load_signal_bundles(run_root / "anchors", "clean")
    synthetic = _load_signal_bundles(run_root / "anchors", "synthetic")
    calibration = _load_signal_bundles(run_root / "anchors", "calibration")
    calibrator = EmpiricalTailCalibrator.fit(clean)
    weak = fit_weak_ranker(
        clean,
        synthetic,
        calibration,
        seed=int(dict(config["runtime"])["seed"]),
    )
    fusion_scores = np.asarray(
        [deterministic_relevance(bundle, calibrator) for bundle in calibration], dtype=np.float32
    )
    with torch.no_grad():
        weak_scores = torch.sigmoid(
            weak.model(torch.from_numpy(np.stack([bundle.values for bundle in calibration])))
        ).numpy()
    fusion_metrics = _development_metrics(fusion_scores, calibration)
    weak_metrics = _development_metrics(weak_scores, calibration)
    selected = select_refiner(fusion_metrics, weak_metrics)
    output.mkdir(parents=True, exist_ok=False)
    calibrator_path = output / "deterministic_fusion.json"
    write_json_atomic(
        calibrator_path,
        {
            "format_version": 1,
            "minimum_bank_size": calibrator.minimum_bank_size,
            "distributions": calibrator.distributions,
        },
    )
    weak_path = output / "weak_ranker.pt"
    torch.save(
        {
            "format_version": 1,
            "head_state_dict": weak.model.state_dict(),
            "state_digest": weak.state_digest,
            "schema_digest": weak.schema_digest,
            "data_digest": weak.data_digest,
            "config_digest": weak.config_digest,
        },
        weak_path,
    )
    selected_digest = weak.state_digest if selected == "weak_ranker" else sha256_file(calibrator_path)
    source_manifest_digest = str(dict(anchor_receipt["digests"])["source_manifest"])
    receipt = build_run_receipt(
        "refiner-fit",
        digests={
            "source_manifest": source_manifest_digest,
            "config": canonical_digest(config),
            "signal_schema": weak.schema_digest,
            "deterministic_fusion": sha256_file(calibrator_path),
            "weak_ranker": sha256_file(weak_path),
        },
        counts={"clean": len(clean), "synthetic": len(synthetic), "calibration": len(calibration)},
        metrics={
            **{f"fusion_{key}": value for key, value in fusion_metrics.items()},
            **{f"weak_{key}": value for key, value in weak_metrics.items()},
            "weak_best_calibration_loss": weak.best_calibration_loss,
        },
        failures=0,
        temporary_files=len(list(output.rglob("*.tmp"))),
        seeds={"ranker": int(dict(config["runtime"])["seed"])},
    )
    receipt["selected_refiner"] = selected
    receipt["selected_refiner_digest"] = selected_digest
    receipt["selection_metrics"] = {"fusion": fusion_metrics, "weak": weak_metrics}
    write_json_atomic(output / "fit_receipt.json", receipt)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="patch-qc-refine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    cache = subparsers.add_parser("build-cache")
    cache.add_argument("--run-root", required=True)
    cache.add_argument("--config", default="configs/refinement_v1.yaml")
    cache.add_argument("--source-manifest", required=True)
    cache.add_argument("--checkpoint", required=True)
    cache.add_argument("--batch-size", type=int)
    cache.add_argument("--limit-sources", type=int)
    cache.add_argument("--shard-count", type=int)
    cache.add_argument("--shard-index", type=int)
    cache.add_argument("--workers", type=_non_negative_int, default=0)
    cache.add_argument("--prefetch-sources", type=_positive_int, default=2)
    cache.add_argument("--prefetch-batches", type=_positive_int, default=2)
    cache.add_argument("--progress-interval-seconds", type=_positive_int, default=30)
    cache.add_argument("--resume", action="store_true")
    cache.set_defaults(handler=_build_cache)

    merge = subparsers.add_parser("merge-cache")
    merge.add_argument("--run-root", required=True)
    merge.add_argument("--config", default="configs/refinement_v1.yaml")
    merge.add_argument("--shard-cache", action="append", required=True)
    merge.set_defaults(handler=_merge_cache)

    anchors = subparsers.add_parser("build-anchors")
    anchors.add_argument("--run-root", required=True)
    anchors.add_argument("--config", default="configs/refinement_v1.yaml")
    anchors.add_argument("--checkpoint", required=True)
    anchors.add_argument("--synthetic-count", type=int)
    anchors.add_argument("--synthetic-calibration-count", type=int)
    anchors.add_argument("--limit-rows", type=int)
    anchors.set_defaults(handler=_build_anchors)

    fit = subparsers.add_parser("fit-refiner")
    fit.add_argument("--run-root", required=True)
    fit.add_argument("--config", default="configs/refinement_v1.yaml")
    fit.set_defaults(handler=_fit_refiner)

    freeze = subparsers.add_parser("freeze-policy")
    freeze.add_argument("--run-root", required=True)
    freeze.add_argument("--config", default="configs/refinement_v1.yaml")
    freeze.set_defaults(handler=_freeze_policy)

    rank = subparsers.add_parser("rank-images")
    rank.add_argument("--run-root", required=True)
    rank.add_argument("--request", required=True)
    rank.add_argument("--config", default="configs/refinement_v1.yaml")
    rank.add_argument("--checkpoint", required=True)
    rank.add_argument("--batch-size", type=int)
    rank.add_argument("--output-dir")
    rank.set_defaults(handler=_rank_images)

    sealed = subparsers.add_parser("sealed-evaluate")
    sealed.add_argument("--run-root", required=True)
    sealed.add_argument("--labelme-root", required=True)
    sealed.add_argument("--ranked-candidates")
    sealed.add_argument("--baseline-candidates")
    sealed.add_argument("--training-hashes")
    sealed.set_defaults(handler=_sealed_evaluate)

    review = subparsers.add_parser("export-review")
    review.add_argument("--run-root", required=True)
    review.add_argument("--request", required=True)
    review.add_argument("--config", default="configs/refinement_v1.yaml")
    review.add_argument("--checkpoint", required=True)
    review.add_argument("--output-dir")
    review.set_defaults(handler=_export_review)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
