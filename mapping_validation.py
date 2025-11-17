#!/usr/bin/env python3
"""
Mapping Validation (Single File, CSV-friendly)
- Accepts TWO CSVs (source and target), or auto-creates sample CSVs if not provided.
- Profiles tables, suggests best target column for a given source column, and generates a Markdown report.
- Optional sync validation vs previous target snapshot (CSV).

Outputs in ./output:
  - source_export.csv, target_export.csv       # inputs used
  - source_columns.csv, target_columns.csv     # per-column profiles
  - source_profile.json, target_profile.json
  - target_candidate_keys.json
  - mapping_suggestion.json
  - sync_metrics.json (if prev snapshot provided)
  - report.md

USAGE
=====
# (A) Run with your CSVs
python mapping_validation.py \
  --source-csv ./source.csv \
  --target-csv ./target.csv \
  --source-col customer_id

# (B) Let the script create sample CSVs automatically
python mapping_validation.py --source-col customer_id

# (C) Sync validation with a previous target snapshot
python mapping_validation.py \
  --source-csv ./source.csv \
  --target-csv ./target.csv \
  --prev-target-csv ./target_prev.csv \
  --source-col customer_id

DEPENDENCIES
============
- pandas
- numpy
"""

from __future__ import annotations
import argparse
import json
import logging
import hashlib
from pathlib import Path
from itertools import combinations
from collections import defaultdict
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(name)s: %(message)s')
log = logging.getLogger("mapping-validation")

# ---------- Helpers ----------
def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)

def create_sample_csvs(output_dir: str | Path) -> Tuple[str, str, str]:
    """Create minimal sample source/target/prev CSVs to run mapping."""
    out = Path(output_dir)
    ensure_dir(out)
    source_path = out / "source_sample.csv"
    target_path = out / "target_sample.csv"
    prev_target_path = out / "target_prev_sample.csv"

    source_csv = """customer_id,first_name,last_name,email
C001,Anita,Rao,anita.rao@example.com
C002,Rahul,Sharma,rahul.sharma@example.com
C003,Meera,Iyer,meera.iyer@example.com
C004,Vijay,Kumar,vijay.kumar@example.com
C005,Neha,Patel,neha.patel@example.com
"""
    target_csv = """cust_id,full_name,email_address,city
001,Anita Rao,anita.rao@example.com,Bengaluru
002,Rahul Sharma,rahul.sharma@example.com,Pune
003,Meera Iyer,meera.iyer@example.com,Chennai
004,Vijay Kumar,vijay.kumar@example.com,Hyderabad
005,Neha Patel,neha.patel@example.com,Mumbai
"""
    prev_target_csv = """cust_id,full_name,email_address,city
001,Anita Rao,anita.rao@example.com,Bengaluru
002,Rahul Sharma,rahul.sharma@example.com,Pune
003,Meera Iyer,meera.iyer@example.com,Chennai
004,Vijay Kumar,old@example.com,Hyderabad
"""

    source_path.write_text(source_csv, encoding="utf-8")
    target_path.write_text(target_csv, encoding="utf-8")
    prev_target_path.write_text(prev_target_csv, encoding="utf-8")

    log.info(f"Created sample CSVs: {source_path}, {target_path}, {prev_target_path}")
    return str(source_path), str(target_path), str(prev_target_path)

# ---------- Profiling ----------
def bucket_dtype(dtype: str) -> str:
    s = str(dtype)
    if 'datetime' in s:
        return 'datetime'
    if 'int' in s or 'float' in s or s in ['int64', 'float64']:
        return 'numeric'
    if 'bool' in s:
        return 'boolean'
    return 'string'

def profile_table(df: pd.DataFrame, sample_rows: int = 2000) -> dict:
    out = {'rows': int(len(df)), 'columns': []}
    for col in df.columns:
        s = df[col]
        dtype_bucket = bucket_dtype(s.dtype)
        nulls = int(s.isna().sum())
        null_ratio = nulls / max(1, len(s))
        if len(s) > sample_rows:
            sample = s.sample(sample_rows, random_state=42)
            distinct = int(sample.nunique(dropna=True))
            uniqueness_ratio = distinct / max(1, len(sample))
        else:
            distinct = int(s.nunique(dropna=True))
            uniqueness_ratio = distinct / max(1, len(s))
        avg_len = float(s.astype(str).str.len().mean()) if dtype_bucket == 'string' else None
        vals = s.dropna().astype(str).head(500)
        patterns = defaultdict(int)
        for v in vals:
            pattern = ''.join(['A' if c.isalpha() else '9' if c.isdigit() else c for c in v])
            patterns[pattern] += 1
        top_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:5]
        out['columns'].append({
            'name': col,
            'dtype': str(s.dtype),
            'dtype_bucket': dtype_bucket,
            'nulls': nulls,
            'null_ratio': round(null_ratio, 6),
            'distinct': distinct,
            'uniqueness_ratio': round(uniqueness_ratio, 6),
            'avg_len': avg_len,
            'top_patterns': top_patterns,
        })
    return out

def detect_candidate_keys(df: pd.DataFrame, unique_threshold: float = 0.95, null_threshold: float = 0.01) -> dict:
    prof = profile_table(df)
    singles = []
    for c in prof['columns']:
        if c['uniqueness_ratio'] >= unique_threshold and c['null_ratio'] <= null_threshold:
            singles.append(c['name'])
    top_cols = [c['name'] for c in sorted(prof['columns'], key=lambda x: x['distinct'], reverse=True)[:6]]
    pairs = []
    for a, b in combinations(top_cols, 2):
        abdf = df[[a, b]].dropna()
        if len(abdf) == 0:
            continue
        ab_uniqueness = abdf.drop_duplicates().shape[0] / max(1, abdf.shape[0])
        if ab_uniqueness >= unique_threshold:
            pairs.append((a, b))
    return {'single_keys': singles, 'pair_keys': pairs, 'profile': prof}

def export_column_profile_csv(profile: dict, path: Path):
    rows = []
    for c in profile['columns']:
        patterns = ';'.join([p for p, cnt in c.get('top_patterns', [])])
        rows.append({
            'name': c['name'],
            'dtype': c['dtype'],
            'bucket': c['dtype_bucket'],
            'null_ratio': c['null_ratio'],
            'distinct': c['distinct'],
            'uniqueness_ratio': c['uniqueness_ratio'],
            'avg_len': c.get('avg_len'),
            'top_patterns': patterns
        })
    pd.DataFrame(rows).to_csv(path, index=False)

# ---------- Mapping ----------
def normalize_series(s: pd.Series, bucket: str) -> pd.Series:
    if bucket == 'string':
        return s.astype(str).str.strip().str.lower().str.replace(r'^0+', '', regex=True)
    if bucket == 'numeric':
        return pd.to_numeric(s, errors='coerce')
    if bucket == 'datetime':
        return pd.to_datetime(s, errors='coerce')
    return s.astype(str)

def name_similarity(a: str, b: str) -> float:
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def type_compatibility(src_dtype: str, tgt_dtype: str) -> float:
    bs = bucket_dtype(src_dtype)
    bt = bucket_dtype(tgt_dtype)
    if bs == bt:
        return 1.0
    if {bs, bt} == {'string', 'numeric'}:
        return 0.6
    return 0.0

def jaccard_overlap(src: pd.Series, tgt: pd.Series, bucket: str, sample_size: int = 2000) -> float:
    s = normalize_series(src.dropna(), bucket)
    t = normalize_series(tgt.dropna(), bucket)
    if len(s) > sample_size:
        s = s.sample(sample_size, random_state=42)
    if len(t) > sample_size:
        t = t.sample(sample_size, random_state=42)
    set_s = set(s.astype(str))
    set_t = set(t.astype(str))
    union = len(set_s | set_t)
    if union == 0:
        return 0.0
    return len(set_s & set_t) / union

def referential_coverage(src: pd.Series, tgt: pd.Series, bucket: str) -> float:
    s = normalize_series(src.dropna(), bucket)
    t_set = set(normalize_series(tgt.dropna(), bucket).astype(str))
    if len(s) == 0:
        return 0.0
    hits = s.astype(str).isin(t_set).sum()
    return hits / len(s)

def score_mapping(src_col: str, tgt_col: str, src: pd.Series, tgt: pd.Series, weights: Dict[str, float]) -> dict:
    src_bucket = bucket_dtype(src.dtype)
    name_score = name_similarity(src_col, tgt_col)
    type_score = type_compatibility(str(src.dtype), str(tgt.dtype))
    overlap_score = jaccard_overlap(src, tgt, src_bucket)
    coverage_score = referential_coverage(src, tgt, src_bucket)
    score = (
        weights['name'] * name_score +
        weights['type'] * type_score +
        weights['overlap'] * overlap_score +
        weights['coverage'] * coverage_score
    )
    return {
        'src_col': src_col,
        'tgt_col': tgt_col,
        'name_score': round(name_score, 4),
        'type_score': round(type_score, 4),
        'overlap_score': round(overlap_score, 4),
        'coverage_score': round(coverage_score, 4),
        'total_score': round(score, 4),
    }

def suggest_best_mapping(src_df: pd.DataFrame, tgt_df: pd.DataFrame, src_col: str, weights: Dict[str, float]) -> dict:
    if src_col not in src_df.columns:
        raise ValueError(f"Source column '{src_col}' not found in source CSV.")
    src_series = src_df[src_col]
    candidates = []
    for tgt_col in tgt_df.columns:
        candidates.append(score_mapping(src_col, tgt_col, src_series, tgt_df[tgt_col], weights=weights))
    candidates.sort(key=lambda x: x['total_score'], reverse=True)
    return {'src_col': src_col, 'best': candidates[0] if candidates else None, 'top3': candidates[:3], 'all': candidates}

# ---------- Sync Validation ----------
def row_hash(df: pd.DataFrame, exclude_cols: Optional[List[str]] = None) -> pd.Series:
    exclude_cols = exclude_cols or []
    cols = [c for c in df.columns if c not in exclude_cols]
    # Create a deterministic string per-row by concatenating stringified column values.
    # This is faster and avoids row-wise Python loops.
    s = df[cols].astype(str).agg('|'.join, axis=1)
    return s.apply(lambda v: hashlib.sha256(v.encode('utf-8')).hexdigest())

def validate_sync(prev_target: pd.DataFrame, curr_target: pd.DataFrame, key_cols: List[str]) -> dict:
    # Build join-keys as deterministic strings from key columns
    src_key_df = prev_target[key_cols].astype(str)
    tgt_key_df = curr_target[key_cols].astype(str)
    src_join = src_key_df.apply(lambda r: '|'.join(r.values), axis=1)
    tgt_join = tgt_key_df.apply(lambda r: '|'.join(r.values), axis=1)

    src_set = set(src_join)
    tgt_set = set(tgt_join)

    inserted = list(tgt_set - src_set)
    deleted = list(src_set - tgt_set)
    common = list(src_set & tgt_set)

    # Select rows that are present in both snapshots, then index them by the join-key
    src_common_mask = src_join.isin(common)
    tgt_common_mask = tgt_join.isin(common)
    src_common = prev_target.loc[src_common_mask].copy()
    tgt_common = curr_target.loc[tgt_common_mask].copy()

    # Align both DataFrames by the join-key so row-wise comparison is meaningful
    src_common.index = src_join[src_common_mask].values
    tgt_common.index = tgt_join[tgt_common_mask].values

    # Use the intersection of columns to avoid mismatches
    common_cols = [c for c in src_common.columns if c in tgt_common.columns]

    # Compute row hashes and align by index
    src_hash = row_hash(src_common[common_cols])
    tgt_hash = row_hash(tgt_common[common_cols])
    tgt_hash = tgt_hash.reindex(src_hash.index)

    updates = int((src_hash != tgt_hash).sum())

    return {
        'counts': {
            'prev_target_rows': int(len(prev_target)),
            'curr_target_rows': int(len(curr_target)),
            'inserted': int(len(inserted)),
            'deleted': int(len(deleted)),
            'updated': updates,
            'matched': int(len(common)),
        },
        'keys': {
            'inserted_keys_sample': inserted[:50],
            'deleted_keys_sample': deleted[:50],
        }
    }

# ---------- Report ----------
def generate_report_md(output_dir: str | Path, report_name: str = 'report.md') -> Path:
    out = Path(output_dir)
    def load(name):
        p = out / name
        return json.loads(p.read_text()) if p.exists() else None

    src_profile = load('source_profile.json')
    tgt_profile = load('target_profile.json')
    tgt_keys = load('target_candidate_keys.json')
    mapping = load('mapping_suggestion.json')
    sync = load('sync_metrics.json')

    md = ["# Data Sync Mapper Report\n"]

    def md_section(title: str):
        md.append(f"\n\n## {title}\n\n")

    def render_profile(profile: dict, label: str):
        md_section(f"Profile: {label}")
        md.append(f"Rows: {profile.get('rows')}\n\n")
        md.append("| Column | DType | Bucket | Null% | Distinct | Unique% | AvgLen | Top Patterns |\n")
        md.append("|---|---|---|---:|---:|---:|---:|---|\n")
        for c in profile['columns']:
            patterns = ', '.join([p for p, cnt in c.get('top_patterns', [])])
            md.append(f"| {c['name']} | {c['dtype']} | {c['dtype_bucket']} | {c['null_ratio']*100:.2f} | "
                      f"{c['distinct']} | {c['uniqueness_ratio']*100:.2f} | {c.get('avg_len') or ''} | {patterns} |\n")

    if src_profile: render_profile(src_profile, 'Source')
    if tgt_profile: render_profile(tgt_profile, 'Target')

    if tgt_keys:
        md_section("Candidate Keys (Target)")
        md.append(f"**Single keys**: {tgt_keys.get('single_keys')}\n\n")
        md.append(f"**Pair keys**: {tgt_keys.get('pair_keys')}\n\n")

    if mapping and mapping.get('best'):
        b = mapping['best']
        md_section("Mapping Suggestion")
        md.append(f"Source column: `{mapping['src_col']}`\n\n")
        md.append(f"**Best target column**: `{b['tgt_col']}` (score={b['total_score']})\n\n")
        md.append("Breakdown:\n\n")
        md.append(f"- Name similarity: {b['name_score']}\n")
        md.append(f"- Type compatibility: {b['type_score']}\n")
        md.append(f"- Value overlap: {b['overlap_score']}\n")
        md.append(f"- Referential coverage: {b['coverage_score']}\n\n")
        md.append("Top 3 candidates:\n\n")
        for r in mapping.get('top3', []):
            md.append(f"- `{r['tgt_col']}` (score={r['total_score']})\n")

    if sync:
        md_section("Sync Validation")
        c = sync.get('counts', {})
        md.append(f"- Previous target rows: {c.get('prev_target_rows')}\n")
        md.append(f"- Current target rows: {c.get('curr_target_rows')}\n")
        md.append(f"- Inserted: {c.get('inserted')}\n")
        md.append(f"- Deleted: {c.get('deleted')}\n")
        md.append(f"- Updated: {c.get('updated')}\n")
        md.append(f"- Matched: {c.get('matched')}\n")

    out_path = out / report_name
    out_path.write_text(''.join(md), encoding='utf-8')
    return out_path

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Mapping Validation (single file, CSV-friendly)")
    p.add_argument('--source-csv', required=False, help='Path to source CSV (if omitted, a sample will be created)')
    p.add_argument('--target-csv', required=False, help='Path to target CSV (if omitted, a sample will be created)')
    p.add_argument('--prev-target-csv', required=False, help='Previous target snapshot CSV (optional)')
    p.add_argument('--source-col', required=True, help='Source column to map to target table')

    # Scoring weights
    p.add_argument('--weight-name', type=float, default=0.30, help='Weight for name similarity')
    p.add_argument('--weight-type', type=float, default=0.25, help='Weight for type compatibility')
    p.add_argument('--weight-overlap', type=float, default=0.30, help='Weight for value overlap')
    p.add_argument('--weight-coverage', type=float, default=0.15, help='Weight for referential coverage')

    # Profiling thresholds
    p.add_argument('--unique-threshold', type=float, default=0.95, help='Uniqueness threshold for keys')
    p.add_argument('--null-threshold', type=float, default=0.01, help='Null ratio threshold for keys')
    p.add_argument('--sample-size', type=int, default=2000, help='Sampling size for overlap computations')

    # Output directory
    p.add_argument('--output-dir', default='./output', help='Output directory')
    return p.parse_args()

def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    out_dir = Path(args.output_dir)

    # If CSVs are missing, create samples
    if not args.source_csv or not args.target_csv:
        src_sample, tgt_sample, prev_sample = create_sample_csvs(out_dir)
        args.source_csv = args.source_csv or src_sample
        args.target_csv = args.target_csv or tgt_sample
        # Only set prev-target-csv if user didn't provide one
        if not args.prev_target_csv:
            args.prev_target_csv = prev_sample
            log.info(f"Using sample previous target snapshot: {args.prev_target_csv}")

    # Read inputs
    src_df = pd.read_csv(args.source_csv)
    tgt_df = pd.read_csv(args.target_csv)

    # Export copies of inputs used
    src_df.to_csv(out_dir / 'source_export.csv', index=False)
    tgt_df.to_csv(out_dir / 'target_export.csv', index=False)

    # Profile both; save JSON and column CSVs
    log.info("Profiling source and target...")
    src_profile = profile_table(src_df, sample_rows=args.sample_size)
    tgt_profile = profile_table(tgt_df, sample_rows=args.sample_size)
    (out_dir / 'source_profile.json').write_text(json.dumps(src_profile, indent=2), encoding='utf-8')
    (out_dir / 'target_profile.json').write_text(json.dumps(tgt_profile, indent=2), encoding='utf-8')
    export_column_profile_csv(src_profile, out_dir / 'source_columns.csv')
    export_column_profile_csv(tgt_profile, out_dir / 'target_columns.csv')
    log.info(f"Column profiles exported: {out_dir / 'source_columns.csv'}, {out_dir / 'target_columns.csv'}")

    # Candidate keys (target)
    log.info("Detecting candidate keys on target...")
    tgt_keys = detect_candidate_keys(tgt_df, unique_threshold=args.unique_threshold, null_threshold=args.null_threshold)
    (out_dir / 'target_candidate_keys.json').write_text(json.dumps(tgt_keys, indent=2), encoding='utf-8')

    # Mapping suggestion
    weights = {
        'name': args.weight_name,
        'type': args.weight_type,
        'overlap': args.weight_overlap,
        'coverage': args.weight_coverage,
    }
    log.info(f"Suggesting best mapping for source column '{args.source_col}' -> target...")
    mapping = suggest_best_mapping(src_df, tgt_df, args.source_col, weights=weights)
    (out_dir / 'mapping_suggestion.json').write_text(json.dumps(mapping, indent=2), encoding='utf-8')
    if mapping.get('best'):
        best = mapping['best']
        log.info(f"Best target column: {best['tgt_col']} (score={best['total_score']})")
    else:
        log.warning("No mapping suggestion found.")

    # Optional sync validation
    if args.prev_target_csv:
        log.info("Validating sync activity vs previous target snapshot...")
        prev_tgt_df = pd.read_csv(args.prev_target_csv)
        if mapping and mapping.get('best'):
            key_cols = [mapping['best']['tgt_col']]
        elif tgt_keys['single_keys']:
            key_cols = [tgt_keys['single_keys'][0]]
        else:
            log.warning("No clear key found; using all columns as key (may be slow/inaccurate).")
            key_cols = list(tgt_df.columns)
        sync_metrics = validate_sync(prev_tgt_df, tgt_df, key_cols)
        (out_dir / 'sync_metrics.json').write_text(json.dumps(sync_metrics, indent=2), encoding='utf-8')
        log.info(f"Sync counts: {sync_metrics['counts']}")

    # Markdown summary
    report_path = generate_report_md(out_dir)
    log.info(f"Report written to {report_path}")
    log.info("Done ✅")

if __name__ == "__main__":
    main()
