from pathlib import Path
import argparse
import csv


def find_report_dirs_without_md(input_dir: Path):
    missing = []
    company_count = 0
    report_count = 0

    for company_dir in sorted(input_dir.iterdir()):
        if not company_dir.is_dir():
            continue

        company_count += 1

        report_dirs = [p for p in sorted(company_dir.iterdir()) if p.is_dir()]
        for report_dir in report_dirs:
            report_count += 1
            has_md = any(report_dir.rglob("*.md"))
            if not has_md:
                missing.append(
                    {
                        "company": company_dir.name,
                        "report_folder": report_dir.name,
                        "report_path": str(report_dir),
                        "report_uri": report_dir.resolve().as_uri(),
                    }
                )

    return missing, company_count, report_count


def write_csv(rows, output_file: Path):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["company", "report_folder", "report_path", "report_uri"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="遍历 pdf2md 下所有公司研报文件夹，统计没有 md 文件的研报目录。"
    )
    parser.add_argument(
        "--input-dir",
        default="pdf2md",
        help="pdf2md 根目录（默认: pdf2md）",
    )
    parser.add_argument(
        "--output-csv",
        default="pdf2md_missing_md_report.csv",
        help="缺失结果导出 CSV 路径（默认: pdf2md_missing_md_report.csv）",
    )
    parser.add_argument(
        "--print-all",
        action="store_true",
        help="在终端打印全部缺失项（默认仅打印前20条）",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在或不是文件夹: {input_dir}")

    missing, company_count, report_count = find_report_dirs_without_md(input_dir)

    print(f"公司文件夹数: {company_count}")
    print(f"研报文件夹总数: {report_count}")
    print(f"无 md 的研报文件夹数: {len(missing)}")

    if missing:
        to_show = missing if args.print_all else missing[:20]
        print("\n缺失示例:")
        for idx, item in enumerate(to_show, 1):
            print(f"{idx}. [{item['company']}] {item['report_folder']} -> {item['report_uri']}")

        if not args.print_all and len(missing) > 20:
            print(f"... 其余 {len(missing) - 20} 条已省略，可加 --print-all 查看全部。")

    output_csv = Path(args.output_csv)
    write_csv(missing, output_csv)
    print(f"\n结果已写入: {output_csv}")


if __name__ == "__main__":
    main()
