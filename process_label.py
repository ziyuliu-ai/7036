
import os


def batch_label_process(input_dir, output_dir):
    """Recursively label all .txt files in `input_dir` and write labeled files to `output_dir`.

    The function preserves directory structure and calls `clean_file` for each
    file. Files are expected to contain one line per logical text segment.
    """

    for root, _, files in os.walk(input_dir):
        for file in files:
            if not file.endswith(".txt"):
                continue

            in_path = os.path.join(root, file)
            rel_path = os.path.relpath(root, input_dir)
            out_dir = os.path.join(output_dir, rel_path)
            os.makedirs(out_dir, exist_ok=True)

            out_path = os.path.join(out_dir, file)
            process_label(in_path, out_path)

def process_label(input_file, output_file):
    """Label a single file and write to output."""
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cleaned_lines = []
    noise_counter = 0

    for line in lines:
        line = line.strip()
        if not line:
            noise_counter += 1
            continue

        tokens = line.lower().split()

        # skip line if labeled as noise
        if tokens and tokens[0] == "noise":
            noise_counter += 1
            continue

        # remove label prefix
        cleaned_line = "".join(tokens[1:]) if len(tokens) > 1 else ""
        if cleaned_line:
            cleaned_lines.append(cleaned_line)


    if len(lines) > 0 and noise_counter/len(lines) < 0.5:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(cleaned_lines))
        print(f"Processed: {input_file} → {output_file}.")
    else:
        print(f"Skipped {input_file} due to excessive noise.")


def main():
    """Entry point: label all cleaned report files into a labeled directory."""
    input_path = r"reports_txt_by_quarter_cleaned_labeled"
    output_path = r"reports_txt_by_quarter_cleaned_done"
    batch_label_process(input_path, output_path)
    print("Processing labels done!")


if __name__ == "__main__":
    main()
