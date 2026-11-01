import pandas as pd

from src.day1.inspect_data import load_dataset

def validate_record(record):

    errors = []

    #Instruction must be non empty strings
    instruction = record["instruction"]

    if not isinstance(instruction, str):
        errors.append("Instruction must be a string")

    elif not instruction.strip():
        errors.append("Instruction cannot be empty")


    #Response must be not empty string
    response = record["response"]

    if not isinstance(response, str):
        errors.append("Response must be a string")

    elif not response.strip():
        errors.append("Response cannot be empty")


    #Context is optional but if provided, it must be a string or none
    context = record["context"]

    if not pd.isna(context) and not isinstance(context, str):
        errors.append("Context must be string or none")


    #Record id must be a non empty string
    record_id = record["record_id"]

    if not isinstance(record_id, str):
        errors.append("Record ID must be string")

    elif not record_id.strip():
        errors.append("Record ID cannot be empty")

    return errors


def valid_dataset(df):

    validation_result = []

    for _, record in df.iterrows():

        errors = validate_record(record)

        validation_result.append({
            "record_id" : record["record_id"],
            "is_valid" : len(errors) == 0,
            "errors" : errors
        })

    return pd.DataFrame(validation_result)


if __name__ == "__main__":

    file_path = "data/raw/synthetic_dirty.jsonl"

    dataset = load_dataset(file_path)

    results = valid_dataset(dataset)

    print("\n---------------VALIDATION RESULTS---------------")

    print("\nFirst 10 Validation Results:")
    print(results.head(10).to_string(index=False))

    print("\nValidation Summary:")

    print("\nTotal Records: ", len(results))

    print("\nValid Records: ", results["is_valid"].sum())

    print("\nInvalid Records: ", len(results) - results["is_valid"].sum())

    print("\n---------------VALIDATION COMPLETED---------------")
