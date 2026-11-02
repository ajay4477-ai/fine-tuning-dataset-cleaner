
import re
import pandas as pd

# Import the dataset loader developed in Day 1
from src.day1.inspect_data import load_dataset


def clean_text(value):
    """
    Clean whitespace from a text value.

    - Remove leading and trailing whitespace.
    - Replace multiple spaces, tabs, and newlines with one space.
    - Preserve non-string values for validation.
    """

    # Do not modify values that are not strings
    if not isinstance(value, str):
        return value

    # Remove whitespace from the beginning and end
    value = value.strip()

    # Replace consecutive whitespace characters with one space
    value = re.sub(r"\s+", " ", value)

    return value


def clean_dataset(df):
    """
    Clean text columns without modifying the original DataFrame.

    Returns:
        pandas.DataFrame: A copy containing cleaned text values.
    """

    # Create a copy to preserve the original raw dataset
    cleaned_df = df.copy()

    # Define the columns that require text cleaning
    text_columns = [
        "instruction",
        "context",
        "response",
        "category",
        "record_id"
    ]

    # Apply the cleaning function to each available text column
    for column in text_columns:

        # Check whether the column exists in the DataFrame
        if column in cleaned_df.columns:

            # Apply clean_text() to every value in the column
            cleaned_df[column] = cleaned_df[column].map(clean_text)

    # Return the fully cleaned DataFrame after processing all columns
    return cleaned_df


if __name__ == "__main__":

    # Define the path to the raw JSONL dataset
    file_path = "data/raw/synthetic_dirty.jsonl"

    # Load the original dataset using the Day 1 loader
    dataset = load_dataset(file_path)

    # Clean the dataset without modifying the original
    cleaned_dataset = clean_dataset(dataset)

    # Display the original dataset
    print("\n--------------- BEFORE CLEANING ---------------")
    print(dataset.head(10).to_string(index=False))

    # Display the cleaned dataset
    print("\n--------------- AFTER CLEANING ---------------")
    print(cleaned_dataset.head(10).to_string(index=False))

    # Display a basic cleaning summary
    print("\n--------------- CLEANING SUMMARY ---------------")

    print("Original Records:", len(dataset))
    print("Cleaned Records:", len(cleaned_dataset))

    print("\nCleaning Completed Successfully!")
