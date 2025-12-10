#!/usr/bin/env python3
"""
task2_extract_parameters.py

Task-2: Extract key blood/lab parameters and numeric values from a cleaned CSV.

Usage:
    python task2_extract_parameters.py --input "C:/path/to/cleaned_dataset.csv" --outdir "C:/path/to/task2_outputs"

Outputs:
 - extracted_parameters_long.csv
 - extracted_parameters_wide.csv
 - extracted_parameters.json
"""
import argparse
import re
from pathlib import Path
import json
import pandas as pd
import numpy as np

# --- Canonical mapping for common parameter synonyms (extend as needed)
PARAM_SYNONYMS = {
    r'\bhemoglobin\b': 'Hemoglobin',
    r'\bhb\b': 'Hemoglobin',
    r'\brbc\b': 'RBC',
    r'\bwbc\b': 'WBC',
    r'\bplatelet\b': 'Platelets',
    r'\bplatelets\b': 'Platelets',
    r'\bhct\b': 'Hematocrit',
    r'\bmcv\b': 'MCV',
    r'\bmch\b': 'MCH',
    r'\bmchc\b': 'MCHC',
    r'\bglucose\b': 'Glucose',
    r'\bblood sugar\b': 'Glucose',
    r'\bfasting glucose\b': 'Glucose (Fasting)',
    r'\bcholesterol\b': 'Cholesterol',
    r'\bhdl\b': 'HDL',
    r'\bldl\b': 'LDL',
    r'\btriglyceride\b': 'Triglycerides',
    r'\bcreatinine\b': 'Creatinine',
    r'\bbun\b': 'BUN',
    r'\balbumin\b': 'Albumin',
    r'\balt\b': 'ALT',
    r'\bast\b': 'AST',
    r'\bvitamin d\b': 'Vitamin D',
    r'\btsh\b': 'TSH',
    r'\bblood pressure\b': 'Blood Pressure'
}
PARAM_PATTERNS = [(re.compile(p, flags=re.I), name) for p, name in PARAM_SYNONYMS.items()]

# Regex to capture "Parameter: value unit" or "Param value unit"
GENERIC_PATTERN = re.compile(
    r'(?P<param>[A-Za-z][A-Za-z0-9 /_\-]{0,40}?)\s*[:\-]\s*(?P<value>[-+]?\d{1,4}(?:[.,]\d{1,4})?(?:\s*e[-+]?\d+)?)\s*(?P<unit>[A-Za-z/%\u00B5µ\.^0-9×\*\-\/\s]*)',
    flags=re.I
)
SHORT_PATTERN = re.compile(
    r'(?P<param>\b[A-Za-z]{1,10}\b)\s+(?P<value>[-+]?\d{1,4}(?:[.,]\d{1,4})?)\s*(?P<unit>[A-Za-z/%\u00B5µ\.^0-9×\*\-\/\s]*)',
    flags=re.I
)

def parse_number(s):
    if s is None: return None
    s = str(s).strip().replace(',', '')
    m = re.search(r'[-+]?\d+(\.\d+)?', s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except:
        return None

def canonical_param(name):
    if name is None: return None
    n = str(name).strip()
    for patt, canon in PARAM_PATTERNS:
        if patt.search(n):
            return canon
    # fallback: normalize whitespace and title case
    return re.sub(r'\s+', ' ', n).strip().title()

def detect_numeric_lab_columns(df, exclude_cols=None, threshold=0.6):
    if exclude_cols is None: exclude_cols = []
    numeric_cols = []
    for c in df.columns:
        if c in exclude_cols:
            continue
        series = df[c].astype(str)
        parsed_mask = series.apply(lambda x: parse_number(x) is not None)
        prop = parsed_mask.sum() / max(1, len(series))
        nunique = series.dropna().nunique()
        if prop >= threshold and nunique > 0 and nunique < len(series)*0.95:
            numeric_cols.append(c)
    return numeric_cols

def extract_from_text(cell_text):
    results = []
    if pd.isna(cell_text) or cell_text is None:
        return results
    s = str(cell_text)
    # generic param: value pattern
    for m in GENERIC_PATTERN.finditer(s):
        param = m.group('param').strip()
        value = parse_number(m.group('value'))
        unit = m.group('unit').strip() if m.group('unit') else ''
        results.append((canonical_param(param), value, unit, m.group(0)))
    # short pattern (Param value unit)
    for m in SHORT_PATTERN.finditer(s):
        param = m.group('param').strip()
        value = parse_number(m.group('value'))
        unit = m.group('unit').strip() if m.group('unit') else ''
        # ignore tiny words that are likely not parameters (e.g., 'Age 30' is OK but 'On 12' is not)
        if len(param) <= 2 and value is None:
            continue
        results.append((canonical_param(param), value, unit, m.group(0)))
    # deduplicate by param+value
    unique = []
    seen = set()
    for r in results:
        key = (r[0], r[1])
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    return unique

def run_extraction(input_csv: Path, outdir: Path, id_col_candidates=None):
    df = pd.read_csv(input_csv, dtype=str).fillna('')
    outdir.mkdir(parents=True, exist_ok=True)

    # pick ID column
    id_col = None
    if id_col_candidates is None:
        id_col_candidates = ['Name', 'Patient ID', 'ID', 'patient_id']
    for c in id_col_candidates:
        if c in df.columns:
            id_col = c
            break
    if id_col is None:
        df['__row_id__'] = df.index.astype(str)
        id_col = '__row_id__'

    exclude = [id_col, 'Date of Admission', 'Discharge Date', 'Doctor', 'Hospital', 'Insurance Provider', 'Admission Type', 'Medication', 'Medical Condition']
    detected_numeric = detect_numeric_lab_columns(df, exclude_cols=exclude, threshold=0.6)

    long_records = []

    # extract from numeric-like columns
    for col in detected_numeric:
        for idx, raw in df[col].astype(str).items():
            val = parse_number(raw)
            if val is not None:
                long_records.append({
                    "patient_id": df.loc[idx, id_col] if id_col in df.columns else idx,
                    "source_column": col,
                    "parameter": canonical_param(col),
                    "value": val,
                    "unit": "",
                    "extracted_text": str(raw)
                })

    # extract from text columns
    text_cols = [c for c in df.columns if c not in detected_numeric]
    for col in text_cols:
        for idx, raw in df[col].astype(str).items():
            if raw.strip() == '':
                continue
            extracted = extract_from_text(raw)
            for param, value, unit, span in extracted:
                long_records.append({
                    "patient_id": df.loc[idx, id_col] if id_col in df.columns else idx,
                    "source_column": col,
                    "parameter": param,
                    "value": value,
                    "unit": unit,
                    "extracted_text": span
                })

    long_df = pd.DataFrame(long_records)
    if 'value' in long_df.columns:
        long_df['value'] = pd.to_numeric(long_df['value'], errors='coerce')

    # pivot to wide
    if not long_df.empty:
        pivot = long_df.sort_values(['patient_id','parameter']).groupby(['patient_id','parameter']).first().reset_index()
        wide = pivot.pivot(index='patient_id', columns='parameter', values='value').reset_index()
    else:
        wide = pd.DataFrame(columns=['patient_id'])

    # save outputs
    long_path = outdir / 'extracted_parameters_long.csv'
    wide_path = outdir / 'extracted_parameters_wide.csv'
    json_path = outdir / 'extracted_parameters.json'

    long_df.to_csv(long_path, index=False)
    wide.to_csv(wide_path, index=False)
    long_df.to_json(json_path, orient='records', force_ascii=False, indent=2)

    return {
        "long_csv": str(long_path),
        "wide_csv": str(wide_path),
        "json": str(json_path),
        "detected_numeric_cols": detected_numeric
    }

def main():
    p = argparse.ArgumentParser(description="Task-2: Extract lab parameters")
    p.add_argument('--input', required=True, help='Path to cleaned_dataset.csv')
    p.add_argument('--outdir', required=True, help='Output folder for task2 outputs')
    args = p.parse_args()

    input_csv = Path(args.input)
    outdir = Path(args.outdir)

    if not input_csv.exists():
        raise SystemExit(f"Input file not found: {input_csv}")

    results = run_extraction(input_csv, outdir)
    print("Extraction complete. Outputs:")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
