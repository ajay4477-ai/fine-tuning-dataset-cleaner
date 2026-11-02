
from pathlib import Path

from src.day1.inspect_data import load_dataset
from src.day2.validate_data import valid_dataset
from src.day3.clean_data import clean_dataset


def separate_records(cleaned_df, validation_results):
    # Separate valid and invalid records using validation results.
    # Invalid records are retained along with their validation errors.

    # Create Boolean masks from the validation results.
    valid_mask = validation_results["is_valid"].to_numpy(dtype=bool)
    invalid_mask = ~valid_mask

    # Separate records without modifying the original DataFrame.
    valid_records = cleaned_df.iloc[valid_mask].copy()
    rejected_records = cleaned_df.iloc[invalid_mask].copy()

    # Attach validation errors to the corresponding rejected records.
    rejected_records["validation_errors"] = (
        validation_results.loc[invalid_mask, "errors"].to_numpy()
    )

    return valid_records, rejected_records


def export_datasets(valid_records, rejected_records, validation_results):
    """
    Export cleaned, rejected, and validation-report datasets.

    Creates output directories if they do not already exist.
    """

    processed_dir = Path("data/processed")
    rejected_dir = Path("data/rejected")
    reports_dir = Path("reports")

    # Create output directories when necessary.
    processed_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Export valid and rejected records in JSONL format.
    valid_records.to_json(
        processed_dir / "cleaned_dataset.jsonl",
        orient="records",
        lines=True,
        force_ascii=False
    )

    rejected_records.to_json(
        rejected_dir / "rejected_records.jsonl",
        orient="records",
        lines=True,
        force_ascii=False
    )

    # Export the record-level validation report.
    validation_results.to_csv(
        reports_dir / "validation_report.csv",
        index=False
    )


if __name__ == "__main__":

    # Define the raw dataset location.
    file_path = "data/raw/synthetic_dirty.jsonl"

    # Step 1: Load the raw dataset.
    dataset = load_dataset(file_path)

    # Step 2: Clean text fields without modifying raw data.
    cleaned_dataset = clean_dataset(dataset)

    # Step 3: Validate the cleaned dataset.
    validation_results = valid_dataset(cleaned_dataset)

    # Step 4: Separate valid and invalid records.
    valid_records, rejected_records = separate_records(
        cleaned_dataset,
        validation_results
    )

    # Step 5: Export the resulting datasets and report.
    export_datasets(
        valid_records,
        rejected_records,
        validation_results
    )

    print("\n========== DATA PREPARATION SUMMARY ==========")

    print("Total Records:", len(dataset))
    print("Valid Records:", len(valid_records))
    print("Rejected Records:", len(rejected_records))

    print("\n========== EXPORTED FILES ==========")

    print("Cleaned Dataset: data/processed/cleaned_dataset.jsonl")
    print("Rejected Dataset: data/rejected/rejected_records.jsonl")
    print("Validation Report: reports/validation_report.csv")

    print("\nData Preparation Pipeline Completed Successfully!")
