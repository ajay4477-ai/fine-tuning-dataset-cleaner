import pandas as pd
from pathlib import Path

def load_dataset(file_path):

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File Not Found {file_path}"
        )

    extension = file_path.suffix.lower()

    if extension == '.csv':
        df = pd.read_csv(file_path)

    elif extension == '.json':
        df = pd.read_json(file_path)

    elif extension == '.jsonl':
        df = pd.read_json(file_path, lines=True)

    else:
        raise ValueError(
            'Unsupported file format'
        )

    return df

def inspect_dataset(df):
    print("\n---------------DATASET iNFORMATION---------------")

    print("\nFirst Five Records:")
    print(df.head())

    print("\nDataset Shape:")
    print(df.shape)

    print("\nColumns Names:")
    print(df.columns.tolist())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nEmpty String Values:")
    print((df == "").sum())

    print("\nDuplicated Records:")
    print(df.astype(str).duplicated().sum())

    print("\nMissing Value Percentage")
    print(df.isnull().mean() * 100)

    print(f"\nTotal Records: {df.shape[0]}")
    print(f"\nTotal Columns: {df.shape[1]}")

    print("\nData Types:")
    print(df.dtypes)

    print("\n---------------INSPECTION COMPLETED---------------")

def get_dataset_summary(df):

    summary = {
        "total_records" : df.shape[0],
        "total_columns" : df.shape[1],
        "duplicate_records": df.astype(str).duplicated().sum(),
        "missing_values" : df.isnull().sum().sum()
    }

    return summary

if __name__== "__main__":

    file_path = "data/raw/synthetic_dirty.jsonl"

    dataset = load_dataset(file_path)

    inspect_dataset(dataset)

    summary = get_dataset_summary(dataset)

    print("\nDataset Summary:")
    print(summary)
