import os
import shutil
import re

def is_translation_failed(text: str) -> bool:
    """
    Detect if a file has failed translation by checking for:
    - High frequency of "I'm sorry" / "I don't know"
    - Excessively long words (> 50 characters)
    - Repeated character or word patterns
    - High proportion of non-English symbols
    """
    text_lower = text.lower()

    # Rule 1: Repeated phrases indicating failed translation
    if text_lower.count("i'm sorry") >= 5 or text_lower.count("i don't know") >= 5:
        return True

    # Rule 2: Excessively long words
    for word in text_lower.split():
        if len(word) > 50:
            return True

    # Rule 3: Repeated character patterns (e.g., "linealinealinealine" or "------")
    if re.search(r'(.)\1{10,}', text_lower):  # Consecutive repeats >= 10
        return True
    if re.search(r'([a-z]+)\1{5,}', text_lower):  # Repeated word pattern
        return True

    # Rule 4 (disabled): Check proportion of non-English symbols
    # total_len = len(text_lower)
    # if total_len > 0:
    #     non_alpha = sum(1 for c in text_lower if not c.isalpha() and not c.isspace())
    #     if non_alpha / total_len > 0.5:  # More than half are symbols
    #         return True

    return False


def check_and_move_files(input_dir: str, bad_dir: str):
    """
    Check all txt files in input_dir and move files detected as failed translations
    to bad_dir while preserving the directory structure.
    """
    os.makedirs(bad_dir, exist_ok=True)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)

                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                if is_translation_failed(content):
                    rel_path = os.path.relpath(file_path, input_dir)
                    target_path = os.path.join(bad_dir, rel_path)
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)

                    print(f"❌ Translation failed: {file_path} → Moving to {target_path}")
                    shutil.move(file_path, target_path)
                else:
                    print(f"✅ OK: {file_path}")

def main():
    input_dir = "reports_txt_by_quarter_cleaned_en"
    bad_dir = "reports_txt_translation_failed"
    check_and_move_files(input_dir, bad_dir)

if __name__ == "__main__":
    main()
