from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

HEADING_RE = re.compile(r"^(#{1,6})[ \t]*(.*?)[ \t]*#*[ \t]*$")
FENCE_RE = re.compile(r"^[ \t]*(```|~~~)")


def normalize_heading(raw: str) -> str:
    return " ".join(raw.strip().split())


def extract_headings(md_path: Path) -> list[str]:
    headings: list[str] = []
    in_fence = False

    try:
        text = md_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return headings

    for line in text.splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        match = HEADING_RE.match(line)
        if not match:
            continue

        heading_text = normalize_heading(match.group(2))
        if heading_text:
            headings.append(heading_text)

    return headings


def count_headings(root: Path) -> Counter[str]:
    counter: Counter[str] = Counter()
    for md_file in root.rglob("*.md"):
        counter.update(extract_headings(md_file))
    return counter


def write_csv(counter: Counter[str], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["heading", "count"])
        for heading, count in sorted(counter.items(), key=lambda x: (-x[1], x[0])):
            writer.writerow([heading, count])


def main() -> None:
    parser = argparse.ArgumentParser(description="统计目录下所有 Markdown 标题出现次数")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("pdf2md"),
        help="要扫描的目录（默认: pdf2md）",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("md_heading_counts.csv"),
        help="统计结果 CSV 输出路径（默认: md_heading_counts.csv）",
    )

    args = parser.parse_args()
    input_dir: Path = args.input_dir

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"输入目录不存在或不是目录: {input_dir}")

    counter = count_headings(input_dir)
    write_csv(counter, args.output_csv)

    total_unique = len(counter)
    total_occurrences = sum(counter.values())

    print(f"扫描目录: {input_dir}")
    print(f"不同标题数量: {total_unique}")
    print(f"标题总出现次数: {total_occurrences}")
    print(f"结果已保存到: {args.output_csv}")
    print("\nTop 30 标题:")

    for heading, count in sorted(counter.items(), key=lambda x: (-x[1], x[0]))[:30]:
        print(f"{count:>6}  {heading}")


if __name__ == "__main__":
    main()
