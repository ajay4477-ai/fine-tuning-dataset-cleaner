import pandas as pd
from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    field_validator
)

from src.day1.inspect_data import load_dataset
from src.day3.clean_data import clean_dataset


class DatasetRecord(BaseModel):
    """
    Pydantic schema for our fine-tuning dataset.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True
    )

    instruction: str
    context: str | None = None
    response: str
    category: str
    record_id: str

    @field_validator(
        "instruction",
        "response",
        "record_id",
        "category"
    )
    @classmethod
    def validate_required_text(cls, value):

        if not value:
            raise ValueError(
                "This field cannot be empty."
            )

        return value


def validate_record(record):
    """
    Validate one dataset record using Pydantic.

    Returns:
        A tuple containing validation status,
        validated data, and validation errors.
    """

    try:

        validated_record = DatasetRecord.model_validate(record)

        return (
            True,
            validated_record.model_dump(),
            []
        )

    except ValidationError as error:

        errors = [
            f"{item['loc'][0]}: {item['msg']}"
            for item in error.errors()
        ]

        return False, None, errors


def validate_dataset(df):
    """
    Validate all records in a Pandas DataFrame using Pydantic.

    Returns:
        A DataFrame containing record IDs,
        validation status, and validation errors.
    """

    validation_results = []

    records = df.to_dict(orient="records")

    for record in records:

        # Convert missing optional context values to None.
        context = record.get("context")

        if (
            pd.api.types.is_scalar(context)
            and pd.isna(context)
        ):
            record["context"] = None

        # Validate the current record.
        is_valid, validated_data, errors = validate_record(
            record
        )

        validation_results.append({
            "record_id": record.get("record_id"),
            "is_valid": is_valid,
            "errors": errors
        })

    return pd.DataFrame(validation_results)


def export_validation_report(results_df):

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    output_file = reports_dir / "pydantic_validation_report.csv"

    results_df.to_csv(output_file, index=False)

    print("\n========== VALIDATION REPORT EXPORT ==========")
    print("Report saved successfully!")
    print("File location:", output_file)
    print("Total records:", len(results_df))



if __name__ == "__main__":

    # Load the raw dataset.
    file_path = "data/raw/synthetic_dirty.jsonl"

    dataset = load_dataset(file_path)

    # Clean the dataset.
    cleaned_dataset = clean_dataset(dataset)

    # Validate all records.
    validation_results = validate_dataset(cleaned_dataset)

    print("\n========== PYDANTIC DATASET VALIDATION ==========")

    print("Total Records:", len(dataset))

    print(
        "Valid Records:",
        validation_results["is_valid"].sum()
    )

    print(
        "Invalid Records:",
        (~validation_results["is_valid"]).sum()
    )

    print("\n========== SAMPLE VALIDATION RESULTS ==========")

    print(
        validation_results.head(10).to_string(index=False)
    )

    # Display invalid records.
    print("\n========== INVALID RECORD DETAILS ==========")

    invalid_records = validation_results[
        ~validation_results["is_valid"]
    ]

    export_validation_report(validation_results)

    print(invalid_records.to_string(index=False))

    print("\nPydantic Dataset Validation Completed!")
