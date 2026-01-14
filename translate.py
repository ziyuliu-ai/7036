import os
from pathlib import Path
import re
import torch
from transformers import pipeline
from datasets import Dataset

def remove_chinese(text: str) -> str:
    """Remove residual Chinese characters from translated text."""
    return re.sub(r'[\u4e00-\u9fff]+', '', text)

def pre_clean(text: str) -> str:
    """Optional: simple pre-translation cleaning to remove extreme symbols and avoid tokenizer errors.
    
    Removes control characters while preserving common punctuation and alphanumerics.
    """
    # Remove control characters while preserving punctuation and English/digits
    text = re.sub(r'[\x00-\x08\x0B-\x1F\x7F]', ' ', text)
    return text

def batch_translate(input_dir: str, output_dir: str, merge_lines: int = 3, batch_size: int = 16, max_length: int = 512):
    """Batch-translate Chinese text files to English using the Helsinki-NLP OPUS model.
    
    Processes all .txt files in input_dir recursively, merges lines into paragraphs,
    translates using GPU (if available) or CPU, and writes results to output_dir
    preserving directory structure. Skips files that already exist.
    
    Args:
        input_dir (str): Root directory containing Chinese text files (.txt).
        output_dir (str): Root directory where English translations will be written.
        merge_lines (int): Number of lines to merge into a single paragraph before translation. Default 3.
        batch_size (int): Batch size for GPU/CPU processing. Default 16.
        max_length (int): Maximum token length for translation. Default 512.
    """
    # Automatically select device: GPU if available, otherwise CPU
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

        # Pre-cleaning (optional)
        paragraphs = [pre_clean(p) for p in paragraphs]

        # Wrap all paragraphs in a Dataset for batch processing
        dataset = Dataset.from_dict({"text": paragraphs})


        # Apply batch translation (true batched processing)
        dataset = dataset.map(translate_batch, batched=True, batch_size=batch_size)

        translated_paragraphs = dataset["translation"]

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(translated_paragraphs))

    print("All translations complete!")

    # Batch translation function
    def translate_batch(batch):
        texts = batch["text"]
        # Mark empty lines to preserve them as-is
        mask_empty = [not t.strip() for t in texts]
        to_translate = [t[:max_length] if t else "" for t in texts]

        # Translate non-empty texts
        try:
            results = translator(
                to_translate,
                batch_size=batch_size,
                truncation=True,      # Let tokenizer truncate to avoid length errors
                max_length=max_length # Control generation length
            )
            translated = [remove_chinese(r["translation_text"]) if r and "translation_text" in r else "" for r in results]
        except Exception as e:
            print(f"Batch translation failed: {e}")
            translated = to_translate  # Keep original text on error

        # Restore empty lines to empty strings
        for i, is_empty in enumerate(mask_empty):
            if is_empty:
                translated[i] = ""

            return {"translation": translated}

if __name__ == "__main__":
    # Directory containing cleaned Chinese research reports
    input_dir = "reports_txt_by_quarter_cleaned_done"
    
    # Directory where English translations will be written
    output_dir = "reports_txt_by_quarter_cleaned_en"
    batch_translate(input_dir, output_dir, merge_lines=3, batch_size=16, max_length=512)
0