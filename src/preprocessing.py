import os
import re
import pandas as pd

def preprocess_code(code: str) -> str:
    # Replace tabs with 4 spaces
    code = code.replace("\t", "    ")
    # Replace multiple spaces with a single space
    code = re.sub(r" +", " ", code)
    # Remove trailing spaces from each line
    code = "\n".join(line.rstrip() for line in code.splitlines())
    # Remove multiple blank lines
    code = re.sub(r"\n\s*\n", "\n", code)
    return code  # ✅ return cleaned code


def preprocess_selected_from_csv_nested(input_dir: str, output_dir: str, csv_file: str, id_column: str = "submission_id"):
    """
    Preprocess only selected submissions from a CSV.
    Works recursively through subfolders (e.g., p00001/s001234.java)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_csv(csv_file)
    selected_ids = df[id_column].astype(str).tolist()

    missing_files = []

    # Walk through all subfolders
    all_files = {}
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".java"):
                submission_id = os.path.splitext(file)[0]
                all_files[submission_id] = os.path.join(root, file)

    for sid in selected_ids:
        if sid not in all_files:
            print(f"⚠️ File {sid}.java not found, skipping.")
            missing_files.append(sid)
            continue

        input_path = all_files[sid]
        output_path = os.path.join(output_dir, f"{sid}.java")

        try:
            with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
                raw_code = f.read()

            clean_code = preprocess_code(raw_code)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(clean_code)

            print(f"✅ Preprocessed {sid}.java")
        except Exception as e:
            print(f"❌ Failed to preprocess {sid}.java: {e}")

    # Save missing files log
    if missing_files:
        pd.DataFrame({"missing_submission_id": missing_files}).to_csv("missing_files.csv", index=False)
        print(f"\n📄 Missing files logged in missing_files.csv")
