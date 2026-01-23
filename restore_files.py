import os
import shutil

def restore_files(bad_dir: str, input_dir: str):
    """
    Restore files from bad_dir back to their original paths in input_dir.
    """
    for root, _, files in os.walk(bad_dir):
        for file in files:
            if file.endswith(".txt"):
                bad_path = os.path.join(root, file)
                # Calculate the relative path to maintain the directory structure
                rel_path = os.path.relpath(bad_path, bad_dir)
                target_path = os.path.join(input_dir, rel_path)

                os.makedirs(os.path.dirname(target_path), exist_ok=True)

                # Move the file back to its original location
                print(f"🔄 Restoring: {bad_path} → {target_path}")
                shutil.move(bad_path, target_path)

if __name__ == "__main__":
    # Directory containing files that failed translation
    bad_dir = "reports_txt_translation_failed"
    # Directory containing original cleaned translated files
    input_dir = "reports_txt_by_quarter_cleaned_en"

    restore_files(bad_dir, input_dir)
