from pathlib import Path
import re

import torch
from transformers import pipeline

def remove_chinese(text: str) -> str:
    """Remove residual Chinese characters from translated text."""
    return re.sub(r'[\u4e00-\u9fff]+', '', text)

def pre_clean(text: str) -> str:
    """Optional: simple pre-translation cleaning to remove extreme symbols and avoid tokenizer errors.
    
    Removes control characters while preserving common punctuation and alphanumerics.
    """
    text = re.sub(r'[\x00-\x08\x0B-\x1F\x7F]', ' ', text)
    return text

def translate_batch(texts, translator, batch_size, max_length):
    """Translate a list of text paragraphs."""
    mask_empty = [not t.strip() for t in texts]
    to_translate = [t[:max_length] if t else "" for t in texts]

    try:
        results = translator(
            to_translate,
            batch_size=batch_size,
            truncation=True,
            max_length=max_length
        )
        translated = [
            remove_chinese(r["translation_text"]) if r and "translation_text" in r else ""
            for r in results
        ]
    except Exception as e:
        print(f"Batch translation failed: {e}")
        translated = to_translate  # Keep original text on error

    for i, is_empty in enumerate(mask_empty):
        if is_empty:
            translated[i] = ""

    return translated

def batch_translate(input_dir: str, output_dir: str, merge_lines: int = 3, batch_size: int = 16, max_length: int = 512):
    """Batch-translate Chinese text files to English using the Helsinki-NLP OPUS model."""
    device = 0 if torch.cuda.is_available() else -1
    print("Using device:", "GPU" if device == 0 else "CPU")

    translator = pipeline(
        "translation",
        model="Helsinki-NLP/opus-mt-zh-en",
        device=device
    )

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_paths = list(input_dir.rglob("*.txt"))
    print(f"Found {len(txt_paths)} text files; starting translation.")

    for txt_path in txt_paths:
        rel_path = txt_path.relative_to(input_dir)
        out_path = output_dir / rel_path

        if out_path.exists():
            print(f"Skipped (already exists): {out_path}")
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Translating: {txt_path} → {out_path}")

        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()

        # Merge lines → paragraphs
        paragraphs = []
        buffer = []
        for line in lines:
            if line.strip():
                buffer.append(line.strip())
                if len(buffer) >= merge_lines:
                    paragraphs.append(" ".join(buffer))
                    buffer = []
            else:
                if buffer:
                    paragraphs.append(" ".join(buffer))
                    buffer = []
                paragraphs.append("")  # Preserve blank lines
        if buffer:
            paragraphs.append(" ".join(buffer))

        # Pre-cleaning
        paragraphs = [pre_clean(p) for p in paragraphs]

        translated_paragraphs = []
        for i in range(0, len(paragraphs), batch_size):
            chunk = paragraphs[i:i + batch_size]
            translated_paragraphs.extend(
                translate_batch(chunk, translator, batch_size, max_length)
            )

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(translated_paragraphs))

    print("All translations complete!")

def main():
    input_dir = "old_version/reports_txt_by_day"
    output_dir = "old_version/reports_txt_by_day_en"
    batch_translate(input_dir, output_dir, merge_lines=3, batch_size=16, max_length=512)    

if __name__ == "__main__":
    main()
