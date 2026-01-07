import re
import os

def clean_text(text):
    """Clean report text by removing sections starting from legal/rating disclaimers.

    The function splits input text into lines, iterates them, and stops collecting
    lines once it encounters any disclaimer or rating-related keyword. It also
    collapses multiple blank lines into a single newline and returns the
    stripped result.

    Args:
        text (str): Raw text content of a report.

    Returns:
        str: Cleaned text with trailing sections removed.
    """

    lines = text.split("\n")
    cleaned_lines = []

    # Keywords that indicate legal disclaimers or end-of-report boilerplate
    disclaimer_keywords = [
        "本研究报告仅供", "不构成对任何人的投资建议", "不对任何人因使用本报告", "本报告仅供", "本研究报告", "本报告",
        "任何形式的分享证券投资收益", "承担相应的法律责任", "版权归本公司所有", "责任",
        "未经书面许可", "不得以任何形式翻版", "风险提示", "特此声明", "风险警示", "声明", "特别声明"
    ]

    # Keywords that indicate company/stock rating explanations (also boilerplate)
    rating_keywords = [
        "投资评级说明", "股票评级说明", "风险评级说明", "评级定义",
        "行业投资评级", "评级标准", "投资评级的说明"
    ]

    stop_keywords = disclaimer_keywords + rating_keywords

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # If any stop keyword appears in the line, stop processing further lines
        if any(k in line for k in stop_keywords):
            break

        cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines)
    # Collapse multiple consecutive blank lines into a single newline
    cleaned_text = re.sub(r"\n{2,}", "\n", cleaned_text)

    return cleaned_text.strip()


def clean_file(in_path, out_path):
    """Read a text file, clean its content, and write the cleaned text out.

    Args:
        in_path (str): Path to the input .txt file.
        out_path (str): Path where the cleaned .txt will be written.
    """

    with open(in_path, "r", encoding="utf-8") as f:
        text = f.read()

    cleaned = clean_text(text)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(cleaned)

    print(f"Cleaned: {in_path} → {out_path}")


def batch_clean(input_dir, output_dir):
    """Recursively clean all .txt files in `input_dir` and write to `output_dir`.

    Directory structure under `input_dir` is preserved under `output_dir`.
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
            clean_file(in_path, out_path)

def main():
    # Default directories; adjust if you use different paths
    input_dir = "reports_txt_by_quarter"
    output_dir = "reports_txt_by_quarter_cleaned"

    batch_clean(input_dir, output_dir)


if __name__ == "__main__":
    main()
