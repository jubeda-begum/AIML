#!/usr/bin/env python3
"""
task1_processing.py

Complete pipeline for Task-1:
- Extraction (read CSV)
- Text processing (cleaning / normalization)
- Text analysis (classify test results, flag lab values if reference ranges are provided)
- Output cleaned CSV / JSON and compact analysis JSON

Usage:
    python task1_processing.py \
        --input /mnt/data/healthcare_dataset.csv \
        --outdir /mnt/data/task1_outputs \
        --ref-ranges ref_ranges.json   # optional
"""

import argparse
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional

import pandas as pd
from dateutil import parser as dateparser

# ------------------------------
# Helpers: normalization & classification
# ------------------------------
def proper_name(name: Optional[str]) -> Optional[str]:
    if pd.isna(name) or name is None:
        return None
    s = re.sub(r'\s+', ' ', str(name).strip())
    # title() is simple; preserve typical prefixes
    return s.title()

def standardize_gender(g: Optional[str]) -> Optional[str]:
    if pd.isna(g) or g is None:
        return None
    s = str(g).strip().lower()
    if s in ('m', 'male'):
        return 'Male'
    if s in ('f', 'female'):
        return 'Female'
    return s.title()

def normalize_date(value: Optional[str]) -> Optional[str]:
    if pd.isna(value) or value is None:
        return None
    # Accept Pandas Timestamp, datetime, or string
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.strftime("%Y-%m-%d")
    s = str(value).strip()
    if s == '':
        return None
    # Try dateutil parser with dayfirst fallback
    for dayfirst in (False, True):
        try:
            dt = dateparser.parse(s, dayfirst=dayfirst, fuzzy=True)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            continue
    return s  # fallback: return original

def standardize_text(s: Optional[str]) -> Optional[str]:
    if pd.isna(s) or s is None:
        return None
    return re.sub(r'\s+', ' ', str(s).strip()).title()

def standardize_medication(s: Optional[str]) -> Optional[str]:
    if pd.isna(s) or s is None:
        return None
    return re.sub(r'\s+', ' ', str(s).strip()).title()

def try_numeric(val):
    """Try converting to float, or return None."""
    if pd.isna(val) or val is None or str(val).strip()=='':
        return None
    try:
        return float(str(val).replace(',', ''))
    except Exception:
        return None

def classify_test_result(s: Optional[str]) -> Tuple[Optional[str], Optional[int]]:
    """
    Map textual test-results to a standardized label and status code:
      Normal -> 0
      Abnormal -> 1
      Inconclusive -> 2
    If unknown, return (original/title-cased, 2)
    """
    if pd.isna(s) or s is None:
        return None, None
    text = str(s).strip().lower()
    if 'normal' in text:
        return "Normal", 0
    if 'abnormal' in text or 'abn' in text:
        return "Abnormal", 1
    if 'inconclusive' in text or 'indeterminate' in text or 'uncertain' in text:
        return "Inconclusive", 2
    # keyword heuristics:
    if re.search(r'\b(high|elevated|above|greater|increased)\b', text):
        return "Abnormal", 1
    if re.search(r'\b(low|decreased|below|less)\b', text):
        return "Abnormal", 1
    # if just a numeric value or range without context — mark inconclusive
    if re.search(r'^[\d\.\-\,\s/]+$', text):
        return "Inconclusive", 2
    # default
    return str(s).title(), 2

def flag_lab_value(value: float, ref_range: Dict[str, float]) -> Optional[str]:
    """
    Given a numeric lab value and a reference range dict with keys 'low' and 'high',
    return 'Low'/'Normal'/'High' or None if unable.
    """
    if value is None:
        return None
    low = ref_range.get('low', None)
    high = ref_range.get('high', None)
    if low is not None and value < low:
        return 'Low'
    if high is not None and value > high:
        return 'High'
    # if both present and value between them
    if low is not None and high is not None and low <= value <= high:
        return 'Normal'
    # if partial info, fallback:
    return None

# ------------------------------
# Main processing function
# ------------------------------
def process_dataframe(df: pd.DataFrame, ref_ranges: Dict[str, Dict[str, float]]=None) -> pd.DataFrame:
    """
    Clean the dataframe, add standardized columns, and produce analysis columns.
    ref_ranges: optional mapping like {"Hemoglobin": {"low":12.0,"high":15.0}, ...}
    """
    clean = df.copy()

    # Columnwise cleaning: be defensive (check existence)
    if 'Name' in clean.columns:
        clean['Name'] = clean['Name'].apply(proper_name)

    if 'Age' in clean.columns:
        clean['Age'] = pd.to_numeric(clean['Age'], errors='coerce').astype('Int64')

    if 'Gender' in clean.columns:
        clean['Gender'] = clean['Gender'].apply(standardize_gender)

    if 'Blood Type' in clean.columns:
        clean['Blood Type'] = clean['Blood Type'].apply(lambda x: str(x).strip() if not pd.isna(x) else None)

    if 'Medical Condition' in clean.columns:
        clean['Medical Condition'] = clean['Medical Condition'].apply(standardize_text)

    for date_col in ['Date of Admission', 'Discharge Date']:
        if date_col in clean.columns:
            clean[date_col] = clean[date_col].apply(normalize_date)

    for col in ['Doctor', 'Hospital', 'Insurance Provider']:
        if col in clean.columns:
            clean[col] = clean[col].apply(standardize_text)

    if 'Billing Amount' in clean.columns:
        clean['Billing Amount'] = pd.to_numeric(clean['Billing Amount'], errors='coerce').round(2)

    if 'Room Number' in clean.columns:
        clean['Room Number'] = pd.to_numeric(clean['Room Number'], errors='coerce').astype('Int64')

    if 'Admission Type' in clean.columns:
        clean['Admission Type'] = clean['Admission Type'].apply(lambda x: str(x).strip().title() if not pd.isna(x) else None)

    if 'Medication' in clean.columns:
        clean['Medication'] = clean['Medication'].apply(standardize_medication)

    # Standardize Test Results -> label + status_code
    if 'Test Results' in clean.columns:
        labels = []
        codes = []
        for v in clean['Test Results']:
            lbl, code = classify_test_result(v)
            labels.append(lbl)
            codes.append(code)
        clean['Test_Results_Standard'] = labels
        clean['Status_Code'] = codes

    # Detect possible numeric lab columns (heuristic: many numeric values) and apply flagging if ref_ranges provided
    if ref_ranges:
        for lab_name, range_vals in ref_ranges.items():
            if lab_name in clean.columns:
                # convert values to numeric
                clean[lab_name + '_Numeric'] = clean[lab_name].apply(try_numeric)
                # flag as Low/Normal/High
                clean[lab_name + '_Flag'] = clean[lab_name + '_Numeric'].apply(
                    lambda x, rv=range_vals: flag_lab_value(x, rv)
                )

    return clean

# ------------------------------
# IO & CLI
# ------------------------------
def load_ref_ranges(path: Optional[Path]) -> Dict[str, Dict[str, float]]:
    if path is None:
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # expected format: {"Hemoglobin": {"low": 12.0, "high": 15.0}, ...}
    return data

def save_outputs(clean: pd.DataFrame, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / 'cleaned_dataset.csv'
    json_path = outdir / 'cleaned_dataset.json'
    analysis_path = outdir / 'analysis_output.json'

    clean.to_csv(csv_path, index=False)
    clean.to_json(json_path, orient='records', indent=2, force_ascii=False)

    # Analysis: compact view
    analysis_cols = []
    # choose columns to include in analysis
    for c in ['Name','Age','Gender','Medical Condition','Test_Results_Standard','Status_Code']:
        if c in clean.columns:
            analysis_cols.append(c)
    analysis_df = clean[analysis_cols] if len(analysis_cols)>0 else clean.head(0)
    analysis_df.to_json(analysis_path, orient='records', indent=2, force_ascii=False)

    return csv_path, json_path, analysis_path

def main():
    p = argparse.ArgumentParser(description="Task-1 CSV processing for health diagnostics")
    p.add_argument('--input', type=str, default='/mnt/data/healthcare_dataset.csv', help='Input CSV path')
    p.add_argument('--outdir', type=str, default='/mnt/data/task1_outputs', help='Output directory')
    p.add_argument('--ref-ranges', type=str, default=None, help='Optional JSON with lab reference ranges')
    args = p.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    ref_ranges_path = Path(args.ref_ranges) if args.ref_ranges else None

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    ref_ranges = load_ref_ranges(ref_ranges_path) if ref_ranges_path else {}

    clean = process_dataframe(df, ref_ranges=ref_ranges)
    csv_path, json_path, analysis_path = save_outputs(clean, outdir)

    print("Processing complete.")
    print(f"Cleaned CSV: {csv_path}")
    print(f"Cleaned JSON: {json_path}")
    print(f"Analysis JSON: {analysis_path}")

if __name__ == '__main__':
    main()
