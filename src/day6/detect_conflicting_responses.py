
from pathlib import Path

import pandas as pd

from src.day1.inspect_data import load_dataset
from src.day3.clean_data import clean_dataset

def normalize_text(value):
    """
    Normalize text for comparison.

    - Convert text to lowercase.
    - Remove leading and trailing whitespace.
    - Replace repeated whitespace with a single space.
    """

    if not isinstance(value, str):
        return ""

    return " ".join(value.casefold().split())

def detect_conflicting_responses(df):
    """
    Identify instructions that have multiple distinct responses.

    Returns:
        pandas.DataFrame containing records that need review.
    """

    required_columns = [
        "record_id",
        "instruction",
        "response"
    ]

    # Check required columns.
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Work with a copy so the original DataFrame is unchanged.
    working_df = df[required_columns].copy()

    # Normalize instructions and responses for comparison.
    working_df["normalized_instruction"] = (
        working_df["instruction"].map(normalize_text)
    )

    working_df["normalized_response"] = (
        working_df["response"].map(normalize_text)
    )

    # Exclude records with empty or invalid instructions/responses.
    valid_mask = (
        working_df["normalized_instruction"].ne("")
        & working_df["normalized_response"].ne("")
    )

    valid_records = working_df.loc[valid_mask].copy()

    # Prepare output columns.
    result_columns = [
        "instruction",
        "record_id",
        "response",
        "response_variant_count",
        "conflicting_record_ids"
    ]

    if valid_records.empty:
        return pd.DataFrame(columns=result_columns)

    # Group records by normalized instruction.
    grouped = valid_records.groupby(
        "normalized_instruction"
    )

    conflict_records = []

    for _, group in grouped:

        # Count distinct normalized responses.
        response_count = group[
            "normalized_response"
        ].nunique()

        # Multiple distinct responses indicate a potential conflict.
        if response_count > 1:

            record_ids = group["record_id"].tolist()

            for _, record in group.iterrows():

                conflict_records.append({
                    "instruction": record["instruction"],
                    "record_id": record["record_id"],
                    "response": record["response"],
                    "response_variant_count": response_count,
                    "conflicting_record_ids": ", ".join(
                        str(record_id)
                        for record_id in record_ids
                    )
                })

    return pd.DataFrame(
        conflict_records,
        columns=result_columns
    )

def export_conflict_report(results_df):
    """
    Export conflicting response records to a CSV report.
    """

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    output_file = (
        reports_dir / "conflicting_responses_report.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print("\n========== REPORT EXPORT ==========")
    print("Report saved successfully!")
    print("File location:", output_file)
    print("Flagged records:", len(results_df))

if __name__ == "__main__":

    # Load the original dataset.
    file_path = "data/raw/synthetic_dirty.jsonl"

    dataset = load_dataset(file_path)

    # Clean whitespace before comparing records.
    cleaned_dataset = clean_dataset(dataset)

    # Detect conflicting responses.
    conflicts = detect_conflicting_responses(
        cleaned_dataset
    )

    print("\n========== CONFLICTING RESPONSE DETECTION ==========")

    print("Total Records:", len(cleaned_dataset))
    print("Flagged Records:", len(conflicts))

    if conflicts.empty:

        print("\nNo conflicting responses found.")

    else:

        print("\n========== CONFLICT SUMMARY ==========")

        print(
            "Instructions with conflicts:",
            conflicts["instruction"].nunique()
        )

    print("\n========== SAMPLE CONFLICTS ==========")

    sample_columns = [
        "record_id",
        "instruction",
        "response_variant_count"
    ]

    print(
        conflicts[sample_columns]
        .head(10)
        .to_string(index=False)
    )

    # Export the report.
    export_conflict_report(conflicts)

    print("\nConflicting Response Detection Completed!")