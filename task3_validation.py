#!/usr/bin/env python3
"""
task3_validation.py
Task-3: Data Validation & Standardization Module.

Usage example (Windows):
python task3_validation.py --input "C:/Users/Lenovo/Downloads/archive/task1_outputs/cleaned_dataset.csv" \
  --outdir "C:/Users/Lenovo/Downloads/archive/task3_outputs" \
  --ref-ranges "C:/Users/Lenovo/Downloads/archive/ref_ranges.json" \
  --missing-strategy fill_median

Outputs (saved to outdir):
 - validated_dataset.csv
 - validated_dataset.json
 - validation_report.json
"""
import argparse, json, re
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Any, Dict
import pandas as pd
import numpy as np

DEFAULT_EXPECTED_COLUMNS = [
    "Name", "Age", "Gender", "Blood Type", "Medical Condition",
    "Date of Admission", "Doctor", "Hospital", "Insurance Provider",
    "Billing Amount", "Room Number", "Admission Type", "Discharge Date",
    "Medication", "Test Results"
]

UNIT_NORMALIZATION = {
    r"mg/dl|mg per dl|mg dl": "mg/dL",
    r"g/dl|g per dl|g dl": "g/dL",
    r"iu/l|iu per l": "IU/L",
    r"mmol/l|mmol per l": "mmol/L",
    r"x10\^3/µl|x10\^3/ul|x10\^3/µl": "x10^3/µL",
    r"µg/dl|ug/dl": "µg/dL",
}

def load_ref_ranges(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if path is None or not Path(path).exists():
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def unify_unit(unit: Optional[str]) -> Optional[str]:
    if unit is None: return None
    s = str(unit).strip().lower()
    if s == '': return None
    for patt, norm in UNIT_NORMALIZATION.items():
        if re.search(patt, s):
            return norm
    return s

def proper_name(name: Optional[str]) -> Optional[str]:
    if pd.isna(name) or name is None: return None
    return re.sub(r'\s+', ' ', str(name).strip()).title()

def standardize_gender(g: Optional[str]) -> Optional[str]:
    if pd.isna(g) or g is None: return None
    s = str(g).strip().lower()
    if s in ('m','male'): return 'Male'
    if s in ('f','female'): return 'Female'
    return s.title()

def normalize_date(value: Optional[str]) -> Optional[str]:
    if pd.isna(value) or value is None: return None
    try:
        return pd.to_datetime(value, dayfirst=False, errors='coerce').strftime("%Y-%m-%d")
    except:
        return str(value)

def try_numeric(x):
    if pd.isna(x) or x is None: return None
    s = str(x).strip().replace(',', '')
    m = re.search(r'[-+]?\d+(\.\d+)?', s)
    if not m: return None
    try:
        return float(m.group(0))
    except:
        return None

def validate_schema(df: pd.DataFrame, expected_columns: List[str]) -> dict:
    present = set(df.columns.tolist())
    missing = [c for c in expected_columns if c not in present]
    extra = [c for c in df.columns if c not in expected_columns]
    return {"missing_columns": missing, "extra_columns": extra}

def coerce_and_standardize(df: pd.DataFrame, ref_ranges: Dict[str, Dict[str,Any]], report: Dict[str, Any]) -> pd.DataFrame:
    df = df.copy()

    # Name
    if "Name" in df.columns:
        before = df["Name"].astype(str).copy()
        df["Name"] = df["Name"].apply(proper_name)
        report["coercion_summary"]["name_changes"] = int((before != df["Name"]).sum())

    # Age
    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors='coerce').astype("Int64")
        implaus = df["Age"].apply(lambda x: True if pd.notna(x) and (x<0 or x>120) else False)
        report["coercion_summary"]["implausible_ages_count"] = int(implaus.sum())
        report["coercion_summary"]["implausible_age_indices"] = df[implaus].index.tolist()

    # Gender
    if "Gender" in df.columns:
        before = df["Gender"].astype(str).copy()
        df["Gender"] = df["Gender"].apply(standardize_gender)
        report["coercion_summary"]["gender_standardized_count"] = int((before.astype(str).str.lower() != df["Gender"].astype(str).str.lower()).sum())

    # Dates
    for col in ["Date of Admission", "Discharge Date"]:
        if col in df.columns:
            before = df[col].astype(str).copy()
            df[col] = df[col].apply(normalize_date)
            report["coercion_summary"][f"{col}_normalized_count"] = int((before != df[col].astype(str)).sum())

    # Billing Amount
    if "Billing Amount" in df.columns:
        df["Billing Amount"] = pd.to_numeric(df["Billing Amount"], errors='coerce').round(2)
        report["coercion_summary"]["billing_amount_nulls_after"] = int(df["Billing Amount"].isna().sum())

    # Room Number
    if "Room Number" in df.columns:
        df["Room Number"] = pd.to_numeric(df["Room Number"], errors='coerce').astype("Int64")

    # Title-case text fields
    for col in ["Medication", "Doctor", "Hospital", "Insurance Provider", "Medical Condition", "Admission Type", "Blood Type"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: None if pd.isna(x) else re.sub(r'\s+', ' ', str(x).strip()).title())

    # Test Results tokens
    if "Test Results" in df.columns:
        def _std(x):
            if pd.isna(x): return None
            s = str(x).strip().lower()
            if "normal" in s: return "Normal"
            if "abnormal" in s: return "Abnormal"
            if "inconclusive" in s or "indeterminate" in s: return "Inconclusive"
            return str(x).title()
        df["Test Results"] = df["Test Results"].apply(_std)

    # Reference range checks
    param_flags = {}
    if ref_ranges:
        for param, rr in ref_ranges.items():
            if param in df.columns:
                df[param + "_Numeric"] = df[param].apply(try_numeric)
                def _flag(v, low=rr.get("low"), high=rr.get("high")):
                    if pd.isna(v): return None
                    try:
                        if low is not None and v < low: return "Low"
                        if high is not None and v > high: return "High"
                        if low is not None and high is not None and low <= v <= high: return "Normal"
                    except:
                        return None
                    return None
                df[param + "_Flag"] = df[param + "_Numeric"].apply(_flag)
                param_flags[param] = {"low": rr.get("low"), "high": rr.get("high"), "unit": rr.get("unit"), "flagged_count": int(df[param + "_Flag"].notna().sum())}
    report["param_flags_summary"] = param_flags

    # Duplicates
    before = len(df)
    df = df.drop_duplicates()
    report["duplicates_removed"] = before - len(df)

    return df

def handle_missing(df: pd.DataFrame, strategy: str="drop_row", fill_constant: Any=None) -> dict:
    rep = {}
    if strategy == "drop_row":
        before = len(df)
        df = df.dropna()
        rep["rows_dropped"] = before - len(df)
        return {"df": df, "report": rep}
    if strategy in ("fill_mean", "fill_median"):
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        rep["numeric_cols_filled"] = {}
        for c in numeric_cols:
            val = df[c].mean() if strategy=="fill_mean" else df[c].median()
            cnt = int(df[c].isna().sum())
            df[c].fillna(val, inplace=True)
            rep["numeric_cols_filled"][c] = {"filled_with": float(val) if pd.notna(val) else None, "count": cnt}
        nonnum = [c for c in df.columns if c not in numeric_cols]
        rep["non_numeric_filled_with"] = "Unknown"
        rep["non_numeric_cols"] = {}
        for c in nonnum:
            cnt = int(df[c].isna().sum())
            df[c].fillna("Unknown", inplace=True)
            rep["non_numeric_cols"][c] = cnt
        return {"df": df, "report": rep}
    if strategy == "fill_constant":
        total_missing = int(df.isna().sum().sum())
        df = df.fillna(fill_constant)
        rep["total_missing_filled"] = total_missing
        rep["filled_with"] = fill_constant
        return {"df": df, "report": rep}
    raise ValueError("Unknown missing strategy")

def run_validation(input_csv: Path, outdir: Path, ref_ranges_path: Optional[Path], missing_strategy: str, fill_constant: Any):
    outdir.mkdir(parents=True, exist_ok=True)
    ref_ranges = load_ref_ranges(ref_ranges_path)

    df = pd.read_csv(input_csv)
    report = {
        "input_rows": int(len(df)),
        "input_columns": df.columns.tolist(),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "schema_validation": {},
        "coercion_summary": {},
        "param_flags_summary": {},
        "missing_handling": {},
        "duplicates_removed": 0,
        "final_rows": None,
        "final_columns": None,
        "notes": []
    }

    report["schema_validation"] = validate_schema(df, DEFAULT_EXPECTED_COLUMNS)
    if report["schema_validation"]["missing_columns"]:
        report["notes"].append("Missing expected columns: " + ", ".join(report["schema_validation"]["missing_columns"]))

    df2 = coerce_and_standardize(df, ref_ranges, report)
    mh = handle_missing(df2, strategy=missing_strategy, fill_constant=fill_constant)
    df3 = mh["df"]
    report["missing_handling"] = mh["report"]
    report["final_rows"] = int(len(df3))
    report["final_columns"] = df3.columns.tolist()

    validated_csv = outdir / "validated_dataset.csv"
    validated_json = outdir / "validated_dataset.json"
    report_path = outdir / "validation_report.json"

    df3.to_csv(validated_csv, index=False)
    df3.to_json(validated_json, orient="records", indent=2, force_ascii=False)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return validated_csv, validated_json, report_path

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--outdir', required=True)
    p.add_argument('--ref-ranges', default=None)
    p.add_argument('--missing-strategy', default='fill_median', choices=['drop_row','fill_mean','fill_median','fill_constant'])
    p.add_argument('--fill-constant', default='Unknown')
    args = p.parse_args()

    validated_csv, validated_json, report_path = run_validation(
        Path(args.input), Path(args.outdir), Path(args.ref_ranges) if args.ref_ranges else None,
        args.missing_strategy, args.fill_constant
    )
    print("Validation complete.")
    print("Validated CSV:", validated_csv)
    print("Validation report:", report_path)

if __name__ == '__main__':
    main()
