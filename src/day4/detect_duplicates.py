
import pandas as pd

from src.day1.inspect_data import load_dataset
from src.day3.clean_data import clean_dataset


def detect_exact_duplicates(df):
    """
    Identify exact duplicate records based on their content.

    The record_id column is excluded because different records
    can have different IDs but identical content.

    Records containing non-scalar values are skipped because
    they cannot safely be compared using Pandas duplicated().

    Returns:
        unique_records: Records with the first occurrence retained.
        duplicate_records: Subsequent duplicate occurrences.
        skipped_records: Records with unsupported comparison values.
    """

    # Define the columns used to compare record content.
    comparison_columns = [
        "instruction",
        "context",
        "response",
        "category"
    ]

    # Check that all required comparison columns exist.
    missing_columns = [
        column for column in comparison_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Identify rows where every comparison value is a scalar.
    scalar_mask = (
        df[comparison_columns]
        .apply(
            lambda column: column.map(pd.api.types.is_scalar)
        )
        .all(axis=1)
    )

    # Separate records that cannot safely be compared.
    skipped_records = df.loc[~scalar_mask].copy()

    # Keep only records with comparable values.
    comparable_records = df.loc[scalar_mask].copy()

    # Identify subsequent occurrences of duplicate content.
    duplicate_mask = comparable_records.duplicated(
        subset=comparison_columns,
        keep="first"
    )

    # Separate unique records and duplicate occurrences.
    unique_records = comparable_records.loc[
        ~duplicate_mask
    ].copy()

    duplicate_records = comparable_records.loc[
        duplicate_mask
    ].copy()

    return unique_records, duplicate_records, skipped_records


if __name__ == "__main__":

    # Define the path to the raw JSONL dataset.
    file_path = "data/raw/synthetic_dirty.jsonl"

    # Load the original dataset.
    dataset = load_dataset(file_path)

    # Standardize whitespace before comparing records.
    cleaned_dataset = clean_dataset(dataset)

    # Detect exact duplicate records.
    (
        unique_records,
        duplicate_records,
        skipped_records
    ) = detect_exact_duplicates(cleaned_dataset)

    # Display duplicate detection summary.
    print("\n========== EXACT DUPLICATE DETECTION ==========")

    print("Total Records:", len(cleaned_dataset))
    print("Unique Records:", len(unique_records))
    print("Duplicate Records:", len(duplicate_records))
    print("Skipped Records:", len(skipped_records))

    # Display duplicate record details.
    print("\n========== DUPLICATE RECORD DETAILS ==========")

    if duplicate_records.empty:
        print("No exact duplicate records found.")
    else:
        print(duplicate_records.head(10).to_string(index=False))

    # Display skipped record details.
    print("\n========== SKIPPED RECORD DETAILS ==========")

    if skipped_records.empty:
        print("No records were skipped.")
    else:
        print(skipped_records.to_string(index=False))

    print("\nDuplicate Detection Completed!")
