from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

HEADING_RE = re.compile(r"^(#{1,6})[ \t]*(.*?)[ \t]*#*[ \t]*$")
FENCE_RE = re.compile(r"^[ \t]*(```|~~~)")
DATE_RE = re.compile(r"^(\d{8})[_-]")

PRIORITY_KEYWORDS = (
    "投资要点",
    "要点",
    "总结",
    "结论",
    "核心观点",
    "观点",
    "正文",
    "投资建议",
    "盈利预测",
    "风险提示",
)

BOILERPLATE_KEYWORDS = (
    "重要声明",
    "投资评级说明",
    "分析师承诺",
    "免责声明",
    "法律声明",
    "研究院",
    "销售团队",
    "联系方式",
    "相关研究",
    "附录",
    "评级说明",
)


@dataclass
class Section:
    index: int
    heading: str
    level: int
    lines: list[str]

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()


def normalize_heading(raw: str) -> str:
    return " ".join(raw.strip().split())


def load_heading_counts(csv_path: Path | None) -> dict[str, int]:
    if not csv_path or not csv_path.exists():
        return {}

    result: dict[str, int] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            heading = normalize_heading(row.get("heading", ""))
            if not heading:
                continue
            try:
                count = int(row.get("count", "0"))
            except ValueError:
                count = 0
            result[heading] = count
    return result


def markdown_to_plain_text(text: str) -> str:
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("!") and "](" in line:
            continue
        if line.startswith("<table") or line.startswith("</table"):
            continue
        if line.startswith("<tr") or line.startswith("</tr"):
            continue
        if line.startswith("<td") or line.startswith("</td"):
            continue
        line = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", line)
        line = re.sub(r"^>+\s*", "", line)
        line = re.sub(r"^[-*+]\s+", "", line)
        line = re.sub(r"^\d+[\.)]\s+", "", line)
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def extract_sections(md_path: Path) -> list[Section]:
    try:
        text = md_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    sections: list[Section] = []
    current_heading = "正文"
    current_level = 1
    current_lines: list[str] = []
    in_fence = False

    for line in text.splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        match = HEADING_RE.match(line)
        if match:
            body = "\n".join(current_lines).strip()
            if body:
                sections.append(
                    Section(
                        index=len(sections),
                        heading=current_heading,
                        level=current_level,
                        lines=current_lines.copy(),
                    )
                )
            current_heading = normalize_heading(match.group(2)) or "正文"
            current_level = len(match.group(1))
            current_lines = []
            continue

        current_lines.append(line)

    body = "\n".join(current_lines).strip()
    if body:
        sections.append(
            Section(
                index=len(sections),
                heading=current_heading,
                level=current_level,
                lines=current_lines.copy(),
            )
        )

    return sections


def content_char_count(text: str) -> int:
    compact = re.sub(r"\s+", "", text)
    return len(compact)


def match_keywords(text: str, keywords: tuple[str, ...]) -> bool:
    lower = text.lower()
    return any(k.lower() in lower for k in keywords)


def section_score(section: Section, heading_counts: dict[str, int]) -> tuple[float, int]:
    plain_text = markdown_to_plain_text(section.text)
    info_len = content_char_count(plain_text)
    if info_len <= 20:
        return -99999.0, info_len

    heading = section.heading
    score = float(min(info_len, 4000))

    if match_keywords(heading, PRIORITY_KEYWORDS):
        score += 2400.0

    if match_keywords(heading, BOILERPLATE_KEYWORDS):
        score -= 5000.0

    lines = [ln.strip() for ln in section.text.splitlines() if ln.strip()]
    if lines:
        table_like = sum(
            1
            for ln in lines
            if "<table" in ln.lower() or "</table" in ln.lower() or "<tr" in ln.lower() or "<td" in ln.lower()
        )
        ratio = table_like / len(lines)
        if ratio > 0.5:
            score -= 1700.0

    global_count = heading_counts.get(heading, 0)
    if global_count > 0 and not match_keywords(heading, PRIORITY_KEYWORDS):
        score -= min(1500.0, math.log1p(global_count) * 200.0)

    return score, info_len


def select_key_sections(
    sections: list[Section],
    heading_counts: dict[str, int],
    top_k: int,
) -> list[Section]:
    scored: list[tuple[float, int, Section]] = []
    for section in sections:
        score, info_len = section_score(section, heading_counts)
        scored.append((score, info_len, section))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    selected: list[Section] = []
    for score, info_len, section in scored:
        heading = section.heading
        is_priority = match_keywords(heading, PRIORITY_KEYWORDS)
        if score <= -3000.0:
            continue
        if info_len < 120 and not is_priority:
            continue
        selected.append(section)
        if len(selected) >= top_k:
            break

    if not selected:
        fallback = sorted(sections, key=lambda s: content_char_count(markdown_to_plain_text(s.text)), reverse=True)
        selected = [s for s in fallback if content_char_count(markdown_to_plain_text(s.text)) >= 120][:top_k]

    selected.sort(key=lambda s: s.index)
    return selected


def extract_date_from_name(file_name: str) -> str | None:
    match = DATE_RE.match(file_name)
    if not match:
        return None
    return match.group(1)


def stock_name_from_md_path(md_path: Path, input_root: Path) -> str:
    try:
        relative = md_path.relative_to(input_root)
    except ValueError:
        return "unknown_stock"

    parts = relative.parts
    if len(parts) >= 2:
        return parts[0]
    return "unknown_stock"


def compose_output_text(md_path: Path, chosen: list[Section]) -> str:
    blocks: list[str] = [f"源文件: {md_path.name}", f"提取块数量: {len(chosen)}", ""]

    for idx, section in enumerate(chosen, start=1):
        clean = markdown_to_plain_text(section.text)
        if not clean:
            continue
        blocks.append(f"【块{idx}】{section.heading}")
        blocks.append(clean)
        blocks.append("")

    return "\n".join(blocks).rstrip() + "\n"


def process_all(
    input_dir: Path,
    output_dir: Path,
    heading_counts_csv: Path | None,
    top_k: int,
    overwrite: bool,
) -> tuple[int, int, int]:
    heading_counts = load_heading_counts(heading_counts_csv)

    output_dir.mkdir(parents=True, exist_ok=True)
    unrecognized_log = output_dir / "unrecognized_dates.log"
    unrecognized: list[str] = []

    total = 0
    written = 0

    for md_file in input_dir.rglob("*.md"):
        total += 1
        date_str = extract_date_from_name(md_file.name)
        if not date_str:
            unrecognized.append(str(md_file))
            continue

        stock = stock_name_from_md_path(md_file, input_dir)
        target_dir = output_dir / date_str / stock
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / f"{md_file.stem}.txt"

        if target_file.exists() and not overwrite:
            continue

        sections = extract_sections(md_file)
        if not sections:
            continue

        selected = select_key_sections(sections, heading_counts, top_k=top_k)
        if not selected:
            continue

        text = compose_output_text(md_file, selected)
        target_file.write_text(text, encoding="utf-8")
        written += 1

    if unrecognized:
        unrecognized_log.write_text("\n".join(unrecognized) + "\n", encoding="utf-8")

    return total, written, len(unrecognized)


def main() -> None:
    parser = argparse.ArgumentParser(description="按Markdown标题块提取高信息量内容并按日频输出TXT")
    parser.add_argument("--input-dir", type=Path, default=Path("pdf2md"), help="Markdown根目录")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports_txt_by_day_md"),
        help="输出目录（按日频组织）",
    )
    parser.add_argument(
        "--heading-counts-csv",
        type=Path,
        default=Path("md_heading_counts.csv"),
        help="count_md_headings.py 输出的标题统计CSV",
    )
    parser.add_argument("--top-k", type=int, default=3, help="每篇保留的高信息量块数量")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在输出")

    args = parser.parse_args()

    if not args.input_dir.exists() or not args.input_dir.is_dir():
        raise SystemExit(f"输入目录不存在或不是目录: {args.input_dir}")

    total, written, unrecognized = process_all(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        heading_counts_csv=args.heading_counts_csv,
        top_k=max(1, args.top_k),
        overwrite=args.overwrite,
    )

    print(f"扫描md文件总数: {total}")
    print(f"成功输出txt数量: {written}")
    print(f"日期无法识别数量: {unrecognized}")
    print(f"输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
