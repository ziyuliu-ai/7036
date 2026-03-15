import os
import shutil
import subprocess
import json
import hashlib
from datetime import datetime
from tqdm import tqdm

# 股票文件夹的根目录
stock_root = "Eastmoney_report_pdf_download\\reports_pdf"
# 输出根目录
output_root = "pdf2md"
os.makedirs(output_root, exist_ok=True)
ERROR_LOG_PATH = os.path.join(output_root, "_conversion_errors.json")
TEMP_INPUT_DIR = os.path.join(output_root, "_tmp_convert_inputs")
PATH_LENGTH_THRESHOLD = 240


def has_md_in_dir(dir_path):
    if not (os.path.exists(dir_path) and os.path.isdir(dir_path)):
        return False

    for _, _, files in os.walk(dir_path):
        for file_name in files:
            if file_name.lower().endswith(".md"):
                return True

    return False


def get_effective_base_name(pdf_path, output_path):
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    layout_target = os.path.abspath(
        os.path.join(output_path, base_name, "auto", f"{base_name}_layout.pdf")
    )

    if len(layout_target) < PATH_LENGTH_THRESHOLD:
        return base_name

    parts = base_name.split("_")
    if len(parts) >= 4:
        date_part = parts[0]
        broker_part = parts[1]
        stock_part = parts[2]
        title_part = "_".join(parts[3:])
        short_title = title_part[:20].strip()
        readable_name = f"{date_part}_{broker_part}_{stock_part}_{short_title}"
    else:
        readable_name = base_name[:40].strip()

    invalid_chars = '<>:"/\\|?*'
    for ch in invalid_chars:
        readable_name = readable_name.replace(ch, " ")

    readable_name = " ".join(readable_name.split())
    readable_name = readable_name.rstrip(" .")

    if not readable_name:
        short_hash = hashlib.md5(base_name.encode("utf-8")).hexdigest()[:10]
        readable_name = f"short_{short_hash}"

    return readable_name


def is_pdf_converted(pdf_path, output_path):
    """
    判断单个 PDF 是否已经转换过：
    只有当输出目录存在且包含至少一个 .md 文件时，才算转换成功。
    """
    base_name = get_effective_base_name(pdf_path, output_path)
    # mineru 默认会在 output_path 下生成一个以 PDF 文件名为名的子目录
    pdf_output_dir = os.path.join(output_path, base_name)
    print(f"Checking conversion output: {pdf_output_dir}")

    return has_md_in_dir(pdf_output_dir)



def convert_single_pdf(pdf_path, output_path, output_dir_name, clean_output_dir=False):
    """
    使用 mineru 转换单个 PDF 文件
    """
    if clean_output_dir:
        pdf_output_dir = os.path.join(output_path, output_dir_name)
        if os.path.isdir(pdf_output_dir):
            print(f"⚠️ 检测到缺失 md，先清理旧目录后重转: {pdf_output_dir}")
            shutil.rmtree(pdf_output_dir)

    original_base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    cmd_input_path = pdf_path
    temp_pdf_path = None
    legacy_output_dir = os.path.join(output_path, original_base_name)

    if output_dir_name != original_base_name:
        if os.path.isdir(legacy_output_dir):
            print(f"🧹 清理历史长目录残留: {legacy_output_dir}")
            shutil.rmtree(legacy_output_dir)

        os.makedirs(TEMP_INPUT_DIR, exist_ok=True)
        temp_pdf_path = os.path.join(TEMP_INPUT_DIR, f"{output_dir_name}.pdf")
        shutil.copy2(pdf_path, temp_pdf_path)
        cmd_input_path = temp_pdf_path
        print(f"🪄 路径过长，使用短输出名: {original_base_name} -> {output_dir_name}")

    cmd = [
        "mineru",
        "-p", cmd_input_path,
        "-o", output_path,
        "--backend", "pipeline",
        "--device", "cuda",
    ]
    print("运行命令:", " ".join(cmd))
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, encoding="utf-8", errors="replace")

    if temp_pdf_path and os.path.exists(temp_pdf_path):
        os.remove(temp_pdf_path)

    return result


def append_error_log(error_logs, stock_name, pdf_path, output_path, stage, message, result=None, output_dir_name=None):
    if output_dir_name is None:
        output_dir_name = get_effective_base_name(pdf_path, output_path)
    pdf_output_dir = os.path.join(output_path, output_dir_name)
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "stock": stock_name,
        "pdf_path": pdf_path,
        "output_dir": pdf_output_dir,
        "stage": stage,
        "message": message,
    }

    if result is not None:
        record["returncode"] = result.returncode
        record["stdout_tail"] = (result.stdout or "")[-4000:]
        record["stderr_tail"] = (result.stderr or "")[-4000:]

    error_logs.append(record)


def save_error_logs(error_logs, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(error_logs, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # 获取所有股票文件夹（reports_pdf 下的一级子目录）
    stock_folders = [
        os.path.join(stock_root, name)
        for name in os.listdir(stock_root)
        if os.path.isdir(os.path.join(stock_root, name))
    ]

    error_logs = []
    success_count = 0
    skip_count = 0

    # 全股票进度条
    for stock_path in tqdm(stock_folders, desc="全股票转换进度", unit="stock"):
        stock_name = os.path.basename(stock_path)
        output_path = os.path.join(output_root, stock_name)
        os.makedirs(output_path, exist_ok=True)

        # 获取当前股票文件夹下的所有 PDF
        pdf_files = [
            os.path.join(stock_path, f)
            for f in os.listdir(stock_path)
            if f.lower().endswith(".pdf") and os.path.isfile(os.path.join(stock_path, f))
        ]

        # 单个股票的 PDF 转换进度条
        for pdf_path in tqdm(pdf_files, desc=f"{stock_name} 转换进度", unit="pdf", leave=False):
            if is_pdf_converted(pdf_path, output_path):
                print(f"⏩ 跳过 {os.path.basename(pdf_path)}，已完成转换")
                skip_count += 1
                continue

            base_name = get_effective_base_name(pdf_path, output_path)
            pdf_output_dir = os.path.join(output_path, base_name)
            need_reconvert = os.path.isdir(pdf_output_dir)

            if need_reconvert:
                print(f"🔁 {os.path.basename(pdf_path)} 输出目录存在但未发现 md，执行重转")

            try:
                result = convert_single_pdf(pdf_path, output_path, output_dir_name=base_name, clean_output_dir=need_reconvert)
            except Exception as e:
                append_error_log(
                    error_logs,
                    stock_name,
                    pdf_path,
                    output_path,
                    stage="run_exception",
                    message=str(e),
                    output_dir_name=base_name,
                )
                print(f"❌ 异常: {os.path.basename(pdf_path)} -> {e}")
                continue

            if result.returncode != 0:
                append_error_log(
                    error_logs,
                    stock_name,
                    pdf_path,
                    output_path,
                    stage="run_failed",
                    message="mineru returned non-zero exit code",
                    result=result,
                    output_dir_name=base_name,
                )
                print(f"❌ 转换失败: {os.path.basename(pdf_path)} (returncode={result.returncode})")
                continue

            if not is_pdf_converted(pdf_path, output_path):
                append_error_log(
                    error_logs,
                    stock_name,
                    pdf_path,
                    output_path,
                    stage="post_check_failed",
                    message="mineru finished but no .md found in output directory",
                    result=result,
                    output_dir_name=base_name,
                )
                print(f"⚠️ 转换完成但未找到 md: {os.path.basename(pdf_path)}")
                continue

            success_count += 1

    if error_logs:
        save_error_logs(error_logs, ERROR_LOG_PATH)
        print(f"\n⚠️ 本次共记录错误 {len(error_logs)} 条，日志已保存: {ERROR_LOG_PATH}")
    else:
        print("\n✅ 本次未发现转换错误")

    print("✅ 所有股票报告已处理完成，结果保存在:", output_root)
    print(f"成功: {success_count}，跳过: {skip_count}，错误: {len(error_logs)}")
