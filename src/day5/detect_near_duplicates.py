
from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.day1.inspect_data import load_dataset
from src.day3.clean_data import clean_dataset


RESULT_COLUMNS = [
    "record_id_1",
    "record_id_2",
    "instruction_1",
    "instruction_2",
    "response_1",
    "response_2",
    "similarity_score",
    "match_type"
]


def normalize_text(value):
    """Normalize text for exact comparison."""
    if not isinstance(value, str):
        return ""

    return " ".join(value.casefold().split())


def detect_near_duplicates(
    df,
    threshold=0.70
):
    """
    Detect exact instruction matches and near-duplicate instructions.

    Similarity is calculated using the instruction column.
    Response comparison is used to identify pairs needing review.
    """

    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    required_columns = [
        "record_id",
        "instruction",
        "response"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    valid_mask = df["instruction"].map(
        lambda value: isinstance(value, str)
        and bool(value.strip())
    )

    valid_records = df.loc[valid_mask].copy()

    if len(valid_records) < 2:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    vectorizer = TfidfVectorizer()

    try:
        tfidf_matrix = vectorizer.fit_transform(
            valid_records["instruction"]
        )
    except ValueError as error:
        if "empty vocabulary" in str(error).lower():
            return pd.DataFrame(columns=RESULT_COLUMNS)
        raise

    similarity_matrix = cosine_similarity(tfidf_matrix)

    results = []

    for i in range(len(valid_records)):
        for j in range(i + 1, len(valid_records)):

            record_1 = valid_records.iloc[i]
            record_2 = valid_records.iloc[j]

            score = float(similarity_matrix[i, j])

            if score < threshold:
                continue

            instruction_1 = normalize_text(
                record_1["instruction"]
            )

            instruction_2 = normalize_text(
                record_2["instruction"]
            )

            response_1 = normalize_text(
                record_1["response"]
            )

            response_2 = normalize_text(
                record_2["response"]
            )

            if instruction_1 == instruction_2:

                if response_1 == response_2:
                    match_type = "exact_instruction_match"
                else:
                    match_type = "different_responses_review"

            else:
                match_type = "near_duplicate"

            results.append({
                "record_id_1": record_1["record_id"],
                "record_id_2": record_2["record_id"],
                "instruction_1": record_1["instruction"],
                "instruction_2": record_2["instruction"],
                "response_1": record_1["response"],
                "response_2": record_2["response"],
                "similarity_score": round(score, 4),
                "match_type": match_type
            })

    results_df = pd.DataFrame(
        results,
        columns=RESULT_COLUMNS
    )

    return results_df


def export_near_duplicate_report(results_df):

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    output_file = reports_dir / "near_duplicate_report.csv"

    results_df.to_csv(output_file, index=False)

    print("\n========== REPORT EXPORT ==========")
    print("Report saved successfully!")
    print("File location:", output_file)
    print("Total flagged pairs:", len(results_df))


if __name__ == "__main__":

    file_path = "data/raw/synthetic_dirty.jsonl"

    dataset = load_dataset(file_path)

    cleaned_dataset = clean_dataset(dataset)

    near_duplicates = detect_near_duplicates(
        cleaned_dataset,
        threshold=0.70
    )

    print("\n========== NEAR-DUPLICATE DETECTION ==========")

    print("Total Records:", len(cleaned_dataset))
    print("Flagged Pairs:", len(near_duplicates))

    if near_duplicates.empty:
        print("No matching pairs found.")

    else:
        print("\n========== MATCH TYPE SUMMARY ==========")

        print(
            near_duplicates["match_type"]
            .value_counts()
            .to_string()
        )

        print("\n========== SAMPLE MATCHES ==========")

        print(
            near_duplicates.head(20).to_string(index=False)
        )

    export_near_duplicate_report(near_duplicates)

    print("\nNear-Duplicate Detection Completed!")
