from pathlib import Path
import argparse
import shutil


def find_short_dirs(root: Path):
    return sorted(
        [p for p in root.rglob("*") if p.is_dir() and p.name.startswith("short_")],
        key=lambda p: len(str(p)),
        reverse=True,
    )


def main():
    parser = argparse.ArgumentParser(description="删除 pdf2md 下所有 short_xxx 目录")
    parser.add_argument(
        "--root",
        default="pdf2md",
        help="扫描根目录（默认: pdf2md）",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="实际执行删除（默认仅预览）",
    )

    args = parser.parse_args()
    root = Path(args.root)

    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"目录不存在或不是文件夹: {root}")

    short_dirs = find_short_dirs(root)
    print(f"发现 short_xxx 目录数量: {len(short_dirs)}")

    if not short_dirs:
        return

    for idx, folder in enumerate(short_dirs, 1):
        print(f"{idx}. {folder}")

    if not args.apply:
        print("\n当前为预览模式。确认删除请加 --apply")
        return

    deleted = 0
    failed = []

    for folder in short_dirs:
        try:
            shutil.rmtree(folder)
            deleted += 1
        except Exception as e:
            failed.append((str(folder), str(e)))

    print(f"\n删除完成: {deleted}")
    print(f"删除失败: {len(failed)}")

    if failed:
        print("失败明细:")
        for path, err in failed:
            print(f"- {path}: {err}")


if __name__ == "__main__":
    main()
