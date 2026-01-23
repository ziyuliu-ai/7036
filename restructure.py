import os
import shutil
import re

def get_quarter(date_str):
    """Extract quarter (Q1-Q4) from an 8-digit date string (YYYYMMDD format)."""
    month = int(date_str[4:6])
    year = date_str[:4]
    if 1 <= month <= 3:
        return f"{year}_Q1"
    elif 4 <= month <= 6:
        return f"{year}_Q2"
    elif 7 <= month <= 9:
        return f"{year}_Q3"
    else:
        return f"{year}_Q4"

def normalize_stock_name(name):
    """Remove special characters from stock name, keeping only Chinese characters, letters, and numbers."""
    name = name.strip()
    # Keep only Chinese characters (\u4e00-\u9fa5), English letters (A-Za-z), and digits (0-9)
    name = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9]", "", name)
    return name

def organize_txt_by_quarter_with_depth(txt_dir, output_dir):
    """Organize text files by quarter and stock name, handling deep report subdirectories."""
    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in os.walk(txt_dir):
        for file in files:
            if not file.endswith(".txt"):
                continue

            # Extract date from filename (first 8 characters)
            date_str = file[:8]
            if not date_str.isdigit():
                print(f"Skipping file with unrecognizable date format: {file}")
                continue

            # Determine the quarter based on the date
            quarter_folder = get_quarter(date_str)

            # Check if this file is in a deep report (in-depth analysis) subdirectory
            is_deep = "深度报告" in root or "DeepReports" in root

            # Extract stock name from the directory structure
            if is_deep:
                # For deep reports, get parent directory name as stock name
                stock_name = normalize_stock_name(os.path.basename(os.path.dirname(root)))
            else:
                # For regular files, get current directory name as stock name
                stock_name = normalize_stock_name(os.path.basename(root))
            print("Stock name:", stock_name)
            if not stock_name:
                print(f"Skipping file with unrecognizable stock name: {file}")
                continue

            # Build the target directory path structure
            target_dir = os.path.join(output_dir, quarter_folder, stock_name)
            # Prevent nested deep report folders if stock name doesn't already contain it
            if is_deep and "深度报告" not in stock_name and "DeepReports" not in stock_name:
                target_dir = os.path.join(target_dir, "DeepReports")

            os.makedirs(target_dir, exist_ok=True)

            # Move the file to the target directory
            src = os.path.join(root, file)
            dst = os.path.join(target_dir, file)
            shutil.move(src, dst)

            print(f"✓ Moved: {file} → {target_dir}")

def main():
    """Main entry point: organize text files from source into quarterly stock directories."""
    txt_dir = r"reports_txt"  # Source directory containing original text files
    output_dir = r"reports_txt_by_quarter"  # Destination directory for organized files
    organize_txt_by_quarter_with_depth(txt_dir, output_dir) 

if __name__ == "__main__":
    main()