"""Fail-closed deterministic cohort and provenance builder."""
from __future__ import annotations
import argparse, hashlib, json, os, tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence
from PIL import Image
from .core import canonical_condition

NATIVE_SIZE = (8192, 5464)
FROZEN_MAIN_SHA256 = "cdd2e3edceae7424f230375573398b31325f24bfffd2f1df0545a03a729aaac4"
FROZEN_COMPARISON_SHA256 = "103f3a25e193ab4b9edfce1d7c8e3f38be20409d261035454df7fa6c1a2c62c7"
ROUTING_VERSION = "cohort-routing-v1"

def cohort_routing_key(value: str) -> str | None:
    standard = canonical_condition(value)
    if standard is not None: return standard
    normalized = value.strip()
    if normalized in {"extra_1", "extra_2", "extra_3", "extra_4"}: return normalized
    if normalized.endswith(("_037", "_187")): return "extra_1"
    if normalized.endswith("_038"): return "extra_2"
    return None

def _sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): d.update(b)
    return d.hexdigest()

def _verify(path: Path, expected: str, name: str) -> str:
    actual = _sha256(path)
    if actual != expected: raise ValueError(f"{name} manifest digest mismatch: expected {expected}, got {actual}")
    return actual

def load_capture_manifest(path: Path) -> list[dict]:
    rows = []
    for n, text in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if text.strip():
            row = json.loads(text)
            if not isinstance(row, dict): raise ValueError(f"{path}:{n} is not a JSON object")
            rows.append(row)
    return rows

def flatten_manifest_images(captures: Sequence[dict]) -> list[dict]:
    rows=[]
    for cap in captures:
        for image in cap.get("images", []):
            if not isinstance(image, dict) or not image.get("path"): continue
            cond=image.get("condition", {}); value=cond.get("condition_id") if isinstance(cond,dict) else image.get("condition_id")
            rows.append({"capture_id":cap.get("capture_id"),"product":cap.get("product"),"split":cap.get("split"),"source_label":cap.get("source_label"),"source_path":str(image["path"]),"condition_id":value,"canonical_condition":cohort_routing_key(str(value or ""))})
    return rows

def _order(row: dict, seed: int) -> tuple[str,str]:
    ident=str(row.get("image_sha256") or row.get("source_path") or row.get("capture_id"))
    return hashlib.sha256(f"{seed}:{ident}".encode()).hexdigest(), ident


def _require_identity(row: dict, *, sha_key: str, label: str) -> tuple[str, str]:
    capture_id = row.get("capture_id")
    image_sha256 = row.get(sha_key)
    if not isinstance(capture_id, str) or not capture_id:
        raise ValueError(f"{label} is missing capture identity")
    if not isinstance(image_sha256, str) or len(image_sha256) != 64:
        raise ValueError(f"{label} is missing a 64-character image SHA-256")
    try:
        int(image_sha256, 16)
    except ValueError as error:
        raise ValueError(f"{label} image SHA-256 is not hexadecimal") from error
    return capture_id, image_sha256


def _assert_query_reference_pair(query: dict, reference: dict) -> None:
    query_capture, query_sha = _require_identity(query, sha_key="image_sha256", label="query")
    reference_capture, reference_sha = _require_identity(reference, sha_key="image_sha256", label="reference")
    if reference_capture == query_capture:
        raise ValueError("query/reference capture leakage")
    if reference_sha == query_sha:
        raise ValueError("query/reference SHA leakage")


def audit_cohort_reference_isolation(rows: Sequence[dict]) -> None:
    """Reject direct and global query/reference reuse before scoring or evaluation.

    Evaluation uses the deliberately stronger cohort policy: every query SHA
    and capture identity must be disjoint from the entire normal-reference
    bank, not merely from the references attached to the same query.
    """
    query_shas: set[str] = set()
    query_captures: set[str] = set()
    reference_shas: set[str] = set()
    reference_captures: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("cohort row is not an object")
        query_capture, query_sha = _require_identity(row, sha_key="image_sha256", label="query")
        references = row.get("references")
        if not isinstance(references, list) or not references:
            raise ValueError("cohort query row is missing references")
        query_shas.add(query_sha)
        query_captures.add(query_capture)
        for reference in references:
            if not isinstance(reference, dict):
                raise ValueError("cohort reference is not an object")
            _assert_query_reference_pair(row, reference)
            reference_capture, reference_sha = _require_identity(reference, sha_key="image_sha256", label="reference")
            reference_shas.add(reference_sha)
            reference_captures.add(reference_capture)
    if query_shas & reference_shas:
        raise ValueError("global query/reference SHA leakage")
    if query_captures & reference_captures:
        raise ValueError("global query/reference capture leakage")


def select_references(images: Sequence[dict], query: dict, count: int, seed: int) -> list[dict]:
    candidates=[r for r in images if r.get("split")=="train" and r.get("product")==query.get("product") and r.get("canonical_condition")==query.get("canonical_condition")]
    choices=[]
    for candidate in candidates:
        if candidate.get("capture_id") == query.get("capture_id"):
            continue
        if query.get("image_sha256") is not None and candidate.get("image_sha256") is not None:
            _assert_query_reference_pair(query, candidate)
        choices.append(candidate)
    unique={}
    for r in sorted(choices,key=lambda r:_order(r,seed)): unique.setdefault(r.get("capture_id"),r)
    return sorted(list(unique.values())[:count],key=lambda r:str(r["capture_id"]))

class _Inspector:
    def __init__(self): self.cache={}
    def inspect(self,path:str):
        if path not in self.cache:
            source=Path(path)
            if not source.is_file(): raise ValueError(f"missing source: {source}")
            try:
                with Image.open(source) as image: size=image.size
            except OSError as e: raise ValueError(f"unreadable source: {source}") from e
            if size != NATIVE_SIZE: raise ValueError(f"wrong geometry: {source} is {size}, expected {NATIVE_SIZE}")
            self.cache[path]=(size,_sha256(source))
        return self.cache[path]

class _Index:
    def __init__(self, images, seed, inspector):
        self.images=list(images); self.seed=seed; self.inspector=inspector; self.by_identity=defaultdict(list); self.banks=defaultdict(list)
        for r in self.images:
            ident=(str(r.get("capture_id")),str(r.get("product")),str(r.get("canonical_condition")))
            if r.get("canonical_condition") is not None: self.by_identity[ident].append(r)
            if r.get("split")=="train" and r.get("canonical_condition") is not None: self.banks[(str(r.get("product")),str(r.get("canonical_condition")))].append(r)
        for bank in self.banks.values(): bank.sort(key=lambda r:_order(r,seed))
    def references(self,q,target,minimum,skips):
        if q.get("canonical_condition") is None: raise ValueError("unsupported condition")
        _, query_digest = self.inspector.inspect(q["source_path"])
        q = {**q, "image_sha256": query_digest}
        selected=[]; seen=set()
        for r in self.banks[(str(q.get("product")),str(q.get("canonical_condition")))]:
            capture=r.get("capture_id")
            if capture==q.get("capture_id") or capture in seen: continue
            try: size,digest=self.inspector.inspect(r["source_path"])
            except ValueError: skips["invalid_reference"]+=1; continue
            item=dict(r); item.update(width=size[0],height=size[1],image_sha256=digest); selected.append(item); seen.add(capture)
            _assert_query_reference_pair(q, item)
            if len(selected)==target: break
        if len(selected) < minimum: raise ValueError(f"insufficient references for {q.get('capture_id')}: need {minimum}, got {len(selected)}")
        return sorted(selected,key=lambda r:str(r["capture_id"]))

def _prepare(row,index,reference_count,minimum_references,skips):
    item=dict(row)
    if item.get("canonical_condition") is None: skips["unsupported_condition"]+=1; return None
    try:
        size,digest=index.inspector.inspect(item["source_path"]); item.update(width=size[0],height=size[1],image_sha256=digest); item["_references"]=index.references(item,reference_count,minimum_references,skips); item["reference_count_target"]=reference_count; item["reference_count_actual"]=len(item["_references"])
    except ValueError as e:
        if str(e).startswith("query/reference") or str(e).startswith("global query/reference"):
            raise
        skips[str(e).split(":",1)[0].replace(" ","_")]+=1; return None
    return item

def _round_robin(rows,count,seed,label):
    groups=defaultdict(list)
    for r in rows: groups[str(r["product"])].append(r)
    for values in groups.values(): values.sort(key=lambda r:_order(r,seed))
    out=[]
    while len(out)<count and any(groups.values()):
        for product in sorted(groups):
            if groups[product] and len(out)<count: out.append(groups[product].pop(0))
    if len(out)!=count: raise ValueError(f"{label} expected {count}, got {len(out)}")
    return out

def _materialize(row,cohort):
    out=dict(row); refs=out.pop("_references"); out["cohort"]=cohort
    out["references"]=[{k:r[k] for k in ("capture_id","source_path","image_sha256","product","canonical_condition","width","height")} for r in refs]; out["reference_capture_ids"]=[r["capture_id"] for r in refs]
    return out

def _validate_label(row,root):
    if not row.get("labeled"): return
    label=root/Path(str(row.get("label_path","")).strip()).name
    if not label.is_file(): raise ValueError(f"missing label: {label}")
    if _sha256(label)!=row.get("label_sha256"): raise ValueError(f"label digest mismatch: {label}")
    try: data=json.loads(label.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e: raise ValueError(f"invalid label: {label}") from e
    if (data.get("imageWidth"),data.get("imageHeight")) != NATIVE_SIZE: raise ValueError(f"label geometry: {label}")
    row["label_path"]=str(label)

def _ng_rows(comparison,root,index,reference_count,minimum_references,skips,labeled_only,skip_ineligible=False):
    seen=set(); out=[]
    for source in comparison:
        if source.get("group","ng")!="ng" or (labeled_only and not source.get("labeled")): continue
        base=Path(str(source.get("source_path","")).strip()).name
        if not base or base in seen: raise ValueError(f"duplicate or missing NG basename: {base}")
        seen.add(base); row=dict(source); row.update(capture_id=f"ng/{source.get('case_id')}",split="ng",source_path=str(root/base),canonical_condition=cohort_routing_key(str(source.get("condition_id") or "")))
        if row["canonical_condition"] is None:
            skips["unsupported_condition"] += 1
            continue
        _,actual=index.inspector.inspect(row["source_path"])
        if actual != source.get("image_sha256"): raise ValueError(f"NG image digest mismatch: {base}")
        _validate_label(row,root); prepared=_prepare(row,index,reference_count,minimum_references,skips)
        if prepared is None:
            if skip_ineligible: continue
            raise ValueError(f"unusable frozen NG member: {source.get('case_id')}")
        out.append(prepared)
    return out

def _inputs(main,comparison,expected_main,expected_comparison):
    return {"main_manifest_sha256":_verify(main,expected_main or FROZEN_MAIN_SHA256,"main"),"comparison_manifest_sha256":_verify(comparison,expected_comparison or FROZEN_COMPARISON_SHA256,"comparison")}

def _assert_frozen_query_shas(ok_rows, ng_rows):
    ok_shas={row.get("image_sha256") for row in ok_rows}
    ng_shas=[row.get("image_sha256") for row in ng_rows]
    if any(not isinstance(value,str) or not value for value in ng_shas): raise ValueError("missing frozen NG SHA")
    if len(ng_shas) != len(set(ng_shas)): raise ValueError("duplicate frozen NG SHA")
    if not all(isinstance(value,str) and value for value in ok_shas): raise ValueError("missing frozen OK SHA")
    if ok_shas & set(ng_shas): raise ValueError("cross-group frozen query SHA collision")

def _assert_output_query_shas(rows, label):
    values=[row.get("image_sha256") for row in rows]
    if any(not isinstance(value,str) or not value for value in values) or len(values)!=len(set(values)):
        raise ValueError(f"duplicate or missing {label} query SHA")

def _resolve_frozen_ok(frozen,index):
    route=cohort_routing_key(str(frozen.get("condition_id") or ""))
    ident=(str(frozen.get("capture_id")),str(frozen.get("product")),str(route))
    candidates=index.by_identity.get(ident,[])
    if not candidates: raise ValueError(f"missing frozen OK member: {ident}")
    basename=Path(str(frozen.get("source_path","")).strip()).name
    exact=[r for r in candidates if Path(r["source_path"]).name==basename]
    if len(exact)==1: return exact[0]
    if len(exact)>1: raise ValueError(f"ambiguous frozen OK basename: {basename}")
    matches=[r for r in candidates if index.inspector.inspect(r["source_path"])[1]==frozen.get("image_sha256")]
    if len(matches)!=1: raise ValueError(f"ambiguous or missing frozen OK SHA: {ident}")
    return matches[0]

def build_smoke_cohort(main_manifest,comparison_manifest,ng_root,ok_count=16,ng_count=16,reference_count=3,seed=17,*,expected_main_digest=None,expected_comparison_digest=None):
    main,comparison=Path(main_manifest),Path(comparison_manifest); inputs=_inputs(main,comparison,expected_main_digest,expected_comparison_digest); skips=Counter(); index=_Index(flatten_manifest_images(load_capture_manifest(main)),seed,_Inspector())
    frozen=load_capture_manifest(comparison)
    frozen_ok=[r for r in frozen if r.get("group")=="ok" and canonical_condition(str(r.get("condition_id") or "")) is not None]
    frozen_ng=[r for r in frozen if r.get("group","ng")=="ng" and r.get("labeled")]
    _assert_frozen_query_shas(frozen_ok,frozen_ng)
    ok=[]
    for member in frozen_ok:
        source=_resolve_frozen_ok(member,index); item=_prepare(source,index,reference_count,reference_count,skips)
        if item is None: continue
        if item["image_sha256"]!=member.get("image_sha256"): raise ValueError(f"frozen smoke OK SHA mismatch: {member.get('case_id')}")
        ok.append(item)
    ok=_round_robin(ok,ok_count,seed,"smoke OK")
    ng=_round_robin(_ng_rows(frozen,Path(ng_root),index,reference_count,reference_count,skips,True,True),ng_count,seed,"smoke NG")
    output_ok,output_ng=[_materialize(r,"smoke_ok") for r in ok],[_materialize(r,"smoke_ng") for r in ng]
    _assert_output_query_shas(output_ok+output_ng,"smoke")
    audit_cohort_reference_isolation(output_ok + output_ng)
    return {"ok":output_ok,"ng":output_ng,"counts":{"ok":ok_count,"ng":ng_count},"skips":dict(sorted(skips.items())),"inputs":inputs,"routing_version":ROUTING_VERSION}

def build_full_cohort(main_manifest,comparison_manifest,ng_root,calibration_count=200,reference_count=5,seed=17,*,evaluation_ok_count=200,evaluation_ng_count=88,expected_main_digest=None,expected_comparison_digest=None):
    main,comp_path=Path(main_manifest),Path(comparison_manifest); inputs=_inputs(main,comp_path,expected_main_digest,expected_comparison_digest); skips=Counter(); index=_Index(flatten_manifest_images(load_capture_manifest(main)),seed,_Inspector()); comp=load_capture_manifest(comp_path)
    calibration=_round_robin([x for x in (_prepare(r,index,reference_count,1,skips) for r in index.images if r.get("split")=="calibration") if x],calibration_count,seed,"calibration")
    frozen_ok=[r for r in comp if r.get("group")=="ok"]
    frozen_ng=[r for r in comp if r.get("group","ng")=="ng"]
    _assert_frozen_query_shas(frozen_ok,frozen_ng)
    if len(frozen_ok)!=evaluation_ok_count: raise ValueError(f"evaluation OK expected {evaluation_ok_count}, got {len(frozen_ok)}")
    if len({r.get("image_sha256") for r in frozen_ok})!=len(frozen_ok): raise ValueError("duplicate frozen OK SHA")
    evaluation_ok=[]
    for frozen in frozen_ok:
        source=_resolve_frozen_ok(frozen,index)
        item=_prepare(source,index,reference_count,1,skips)
        if item is None or item["image_sha256"]!=frozen.get("image_sha256"): raise ValueError(f"frozen OK SHA mismatch: {frozen.get('case_id')}")
        evaluation_ok.append(item)
    evaluation_ng=_ng_rows(comp,Path(ng_root),index,reference_count,1,skips,False)
    if len(evaluation_ng)!=evaluation_ng_count: raise ValueError(f"evaluation NG expected {evaluation_ng_count}, got {len(evaluation_ng)}")
    output_calibration,output_ok,output_ng=[_materialize(r,"calibration_ok") for r in calibration],[_materialize(r,"evaluation_ok") for r in evaluation_ok],[_materialize(r,"evaluation_ng") for r in evaluation_ng]
    _assert_output_query_shas(output_calibration+output_ok+output_ng,"full")
    audit_cohort_reference_isolation(output_calibration + output_ok + output_ng)
    return {"calibration":output_calibration,"evaluation_ok":output_ok,"evaluation_ng":output_ng,"counts":{"calibration":calibration_count,"evaluation_ok":evaluation_ok_count,"evaluation_ng":evaluation_ng_count},"skips":dict(sorted(skips.items())),"inputs":inputs,"routing_version":ROUTING_VERSION}

def _atomic_write(path,contents):
    path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(dir=path.parent,prefix=f".{path.name}.",suffix=".tmp")
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as f: f.write(contents); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
    except BaseException: Path(tmp).unlink(missing_ok=True); raise

def _write_jsonl(path,rows):
    text="".join(json.dumps(r,sort_keys=True)+"\n" for r in rows); _atomic_write(path,text); return hashlib.sha256(text.encode()).hexdigest()

def _write_result(result,out,mode):
    groups={"smoke.jsonl":result["ok"]+result["ng"]} if mode=="smoke" else {"calibration.jsonl":result["calibration"],"evaluation_ok.jsonl":result["evaluation_ok"],"evaluation_ng.jsonl":result["evaluation_ng"]}; all_rows=[row for rows in groups.values() for row in rows]; query_shas=[r["image_sha256"] for r in all_rows]; _assert_output_query_shas([{"image_sha256":value} for value in query_shas],mode); audit_cohort_reference_isolation(all_rows); files={n:_write_jsonl(out/n,r) for n,r in groups.items()}; memberships={n:hashlib.sha256("\n".join(sorted(r["image_sha256"] for r in rows)).encode()).hexdigest() for n,rows in groups.items()}; targets=sorted({r["reference_count_target"] for rows in groups.values() for r in rows}); receipt={"mode":mode,"inputs":result["inputs"],"routing_version":result["routing_version"],"reference_count_targets":targets,"global_query_sha_count":len(query_shas),"global_query_sha_set_sha256":hashlib.sha256("\n".join(sorted(query_shas)).encode()).hexdigest(),"counts":result["counts"],"skips":result["skips"],"failures":0,"temporary_files":0,"files":files,"membership_set_sha256":memberships}; _atomic_write(out/"cohort_receipt.json",json.dumps(receipt,sort_keys=True,indent=2)+"\n"); return out/"cohort_receipt.json"

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("mode",choices=("smoke","full"));
    for name in ("main-manifest","comparison-manifest","ng-root","output-dir"): p.add_argument("--"+name,required=True,type=Path)
    p.add_argument("--ok-count",type=int,default=16); p.add_argument("--ng-count",type=int,default=16); p.add_argument("--calibration-count",type=int,default=200); p.add_argument("--reference-count",type=int); p.add_argument("--seed",type=int,default=17); p.add_argument("--expected-main-digest"); p.add_argument("--expected-comparison-digest"); a=p.parse_args(argv)
    try:
        result=build_smoke_cohort(a.main_manifest,a.comparison_manifest,a.ng_root,a.ok_count,a.ng_count,a.reference_count or 3,a.seed,expected_main_digest=a.expected_main_digest,expected_comparison_digest=a.expected_comparison_digest) if a.mode=="smoke" else build_full_cohort(a.main_manifest,a.comparison_manifest,a.ng_root,a.calibration_count,a.reference_count or 5,a.seed,expected_main_digest=a.expected_main_digest,expected_comparison_digest=a.expected_comparison_digest)
        print(_write_result(result,a.output_dir,a.mode)); return 0
    except ValueError as e: print(str(e),file=__import__("sys").stderr); return 2
if __name__=="__main__": raise SystemExit(main())
