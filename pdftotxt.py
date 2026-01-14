import os
import re

import fitz 
from PIL import Image

from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from surya.layout import LayoutPredictor
from surya.settings import settings

# =========================
# 1. Initialize Surya models (default hardware configuration)
# =========================
# Surya is a document layout analysis and OCR library.
# These models will be used for multi-page OCR and layout detection.

foundation = FoundationPredictor()
detector = DetectionPredictor()
recognizer = RecognitionPredictor(foundation)
layout_predictor = LayoutPredictor(
    FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
)

# =========================
# 2. PDF rendering
# =========================

def pdf_to_images_fast(pdf_path, dpi=150):
    """Convert PDF pages to PIL Image objects at the specified DPI.

    Args:
        pdf_path (str): Path to the PDF file.
        dpi (int): Resolution for rendering. Default 150 is a good balance between
                   quality and speed.

    Returns:
        list: List of PIL Image objects, one per page.
    """
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        try:
            pix = page.get_pixmap(dpi=dpi, annots=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            images.append(img)
        except Exception as e:
            print(f"[Warning] Rendering failed, skipping page: {e}")
            continue
    return images

# =========================
# 3. Layout region detection (filter after full-page OCR)
# =========================

def line_in_bbox(line_bbox, region_bbox):
    """Check whether a line bounding box overlaps with a region bounding box."""
    x1, y1, x2, y2 = line_bbox
    rx1, ry1, rx2, ry2 = region_bbox
    return not (x2 < rx1 or x1 > rx2 or y2 < ry1 or y1 > ry2)

def is_in_text_region(line, layout_page):
    """Check whether a line falls within one of the text-like layout regions.

    Text-like regions include "Text", "SectionHeader", and "ListItem" labels.
    """
    for region in layout_page.bboxes:
        if region.label in ["Text", "SectionHeader", "ListItem"]:
            if line_in_bbox(line.bbox, region.bbox):
                return True
    return False

# =========================
# 4. Blank-aware line filtering
# =========================

def is_blank_line(line, threshold=0.85):
    """Check whether a line is mostly blank.

    Uses Surya's blank_ratio attribute if available. Higher threshold values
    are stricter (0.85–0.95 recommended).

    Args:
        line: A detected text line object with optional blank_ratio attribute.
        threshold (float): Ratio above which a line is considered blank.

    Returns:
        bool: True if the line is blank or whitespace-only.
    """
    if hasattr(line, "blank_ratio"):
        return line.blank_ratio >= threshold

    if not line.text or line.text.strip() == "":
        return True

    return False

# =========================
# 5. Table-aware region filtering (key optimization)
# =========================

def in_any_region(line_bbox, region_list):
    """Check whether a line's bounding box falls within any of the given regions.

    This is critical for filtering out lines in tables (e.g., "主要股东 27.55%")
    that would otherwise contaminate body text.
    """
    for (x1, y1, x2, y2) in region_list:
        if not (line_bbox[2] < x1 or line_bbox[0] > x2 or line_bbox[3] < y1 or line_bbox[1] > y2):
            return True
    return False

# =========================
# 6. Text cleaning (basic + strict)
# =========================
# Regular expressions for detecting contact info and cleaning content.

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
PHONE_RE = re.compile(r"\d{3,4}-\d{7,8}")

def remove_html(text: str) -> str:
    """Remove HTML tags from text."""
    return re.sub(r"<[^>]+>", "", text)

def is_contact_info(text: str) -> bool:
    """Check whether text contains contact information (email, phone, URL, analyst name)."""
    if EMAIL_RE.search(text): return True
    if URL_RE.search(text): return True
    if PHONE_RE.search(text): return True
    if any(k in text for k in ["电话", "邮箱", "研究员", "SAC"]): return True
    return False

def is_footnote(text: str) -> bool:
    """Check whether text is a footnote or note."""
    return text.startswith(("注", "备注", "*"))

def is_page_number(text: str) -> bool:
    """Check whether text is a page number."""
    return re.fullmatch(r"\d{1,3}", text) is not None

def is_toc_line(text: str) -> bool:
    """Check whether text looks like a table of contents line."""
    if re.search(r"\.{3,}", text): return True
    if re.search(r"\d+$", text) and " " in text: return True
    return False

def is_header_footer(text: str) -> bool:
    """Check whether text is a typical report header or footer."""
    keywords = ["请务必阅读", "重要声明", "证券研究报告", "研究所"]
    return any(k in text for k in keywords)

def is_fragment(text: str) -> bool:
    """Check whether text is a short, isolated fragment (price, rating label, etc.)."""
    if is_contact_info(text): return True
    if any(k in text for k in ["当前价", "目标价", "买入评级", "维持评级", "上调评级"]): return True
    if re.fullmatch(r"[0-9\.\-]+(元|倍)?", text): return True
    if re.fullmatch(r"\d+(\.\d+)?\s*(元|倍|万股|亿元|万股)", text): return True
    return False

def is_short_fragment(text: str) -> bool:
    """Check whether text is very short (2 characters or less)."""
    return len(text.strip()) <= 2

def is_meta_info(text: str) -> bool:
    """Check whether text is metadata, disclaimer, or source attribution."""
    meta_keywords = [
        "资料来源", "来源", "Wind", "WIND",
        "分析师", "执业证书", "免责声明",
        "请阅读", "重要声明",
        "风险提示",
        "证券有限责任公司",
        "投资评级说明", "行业的投资评级说明",
        "研究发展部",
        "地址:", "网址:", "邮编:",
    ]
    return any(k in text for k in meta_keywords)

def clean_line_basic(line: str):
    """Apply basic cleaning rules (page numbers, TOC, headers, footers)."""
    line = line.strip()
    if not line: return None
    if is_page_number(line): return None
    if is_toc_line(line): return None
    if is_header_footer(line): return None
    return line

def clean_text_strict(text: str):
    """Apply strict cleaning rules (contact info, fragments, meta, HTML)."""
    text = remove_html(text).strip()
    if not text: return None
    if is_footnote(text): return None
    if is_contact_info(text): return None
    if is_fragment(text): return None
    if is_meta_info(text): return None
    if is_short_fragment(text): return None
    return text

# =========================
# 7. Full PDF processing (page OCR → Layout → blank + table + cleaning)
# =========================

def pdf_to_text_fast(pdf_path: str) -> str:
    """Convert a full PDF to clean text using multi-stage filtering.

    Pipeline:
    1. Render PDF pages to images
    2. Run full-page OCR on all images
    3. Detect layout regions on all images
    4. Filter lines by: blank ratio, table regions, layout zones, basic/strict rules
    5. Join remaining lines into output text

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Concatenated clean text from all pages.
    """
    images = pdf_to_images_fast(pdf_path, dpi=150)
    if not images:
        return ""

    # Full-page OCR
    ocrs = recognizer(images, det_predictor=detector)

    # Full-page layout detection
    layouts = layout_predictor(images)

    output_lines = []

    for i in range(len(images)):
        ocr_page = ocrs[i]
        layout_page = layouts[i]

        # Extract table regions (key optimization)
        table_regions = [r.bbox for r in layout_page.bboxes if r.label == "Table"]

        for line in ocr_page.text_lines:

            # 0. blank flitering
            if is_blank_line(line):
                continue

            # 1. table filtering
            if in_any_region(line.bbox, table_regions):
                continue

            raw = line.text
            if not raw:
                continue

            # 2. Layout region filter
            if not is_in_text_region(line, layout_page):
                continue

            # 3. Basic cleaning (page numbers, TOC, headers)
            basic = clean_line_basic(raw)
            if not basic:
                continue

            # 4. Strict cleaning (contact info, fragments, metadata)
            strict = clean_text_strict(basic)
            if not strict:
                continue

            output_lines.append(strict)

    return "\n".join(output_lines)

# =========================
# 8. Batch processing (single-process by default)
# =========================

def build_output_path(pdf_path: str, pdf_dir: str, txt_dir: str) -> str:
    """Construct the output text file path from a PDF path.

    Preserves the stock ticker structure and detects whether a PDF is in the
    DeepReports folder.

    Args:
        pdf_path (str): Full path to input PDF.
        pdf_dir (str): Root PDF directory.
        txt_dir (str): Root output text directory.

    Returns:
        str: Output text file path.
    """
    rel = os.path.relpath(pdf_path, pdf_dir)
    parts = rel.split(os.sep)

    # Check if PDF is in a DeepReports subfolder
    is_deep = "DeepReports" in parts or "深度报告" in parts

    if is_deep:
        # For deep reports: .../stock_name/DeepReports/report.pdf
        deep_label = "DeepReports" if "DeepReports" in parts else "深度报告"
        idx = parts.index(deep_label)
        stock = parts[idx - 1]
        out_dir = os.path.join(txt_dir, stock, "DeepReports")
    else:
        # For regular reports: .../stock_name/report.pdf
        stock = parts[0]
        out_dir = os.path.join(txt_dir, stock)

    os.makedirs(out_dir, exist_ok=True)
    txt_filename = os.path.splitext(os.path.basename(pdf_path))[0] + ".txt"
    return os.path.join(out_dir, txt_filename)


def batch_convert(pdf_dir: str, txt_dir: str):
    """Convert all PDFs in a directory tree to text files.

    Recursively finds all .pdf files under `pdf_dir`, converts each to text
    using `pdf_to_text_fast`, and writes output to `txt_dir` preserving the
    original directory structure. Existing text files are skipped.

    Args:
        pdf_dir (str): Root directory containing PDFs (may have subdirectories).
        txt_dir (str): Root directory where output text files will be written.
    """
    pdf_paths = []
    for root, _, files in os.walk(pdf_dir):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_paths.append(os.path.join(root, file))

    print(f"Found {len(pdf_paths)} PDFs; processing sequentially.")

    for pdf_path in pdf_paths:
        txt_path = build_output_path(pdf_path, pdf_dir, txt_dir)

        # If output text already exists, skip this PDF
        if os.path.exists(txt_path):
            print(f"Skipped (already exists): {txt_path}")
            continue

        print(f"Converting: {pdf_path} → {txt_path}")
        text = pdf_to_text_fast(pdf_path)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(txt_path), exist_ok=True)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

    print("All conversions complete!")

if __name__ == "__main__":
    pdf_root = r"Eastmoney_report_pdf_download\reports_pdf"
    txt_root = r"reports_txt"
    batch_convert(pdf_root, txt_root)
