import sys
import re
import numpy as np
import os
from sentence_transformers import SentenceTransformer, util

# ============================================================
# 1. Load the BGE-m3 embedding model
# ============================================================

class EmbeddingClassifier:
    def __init__(self):
        print("Loading BGE-m3 embedding model...")
        self.model = SentenceTransformer("BAAI/bge-m3")

        # Positive examples (representative body text)
        self.pos_examples = [
            "公司发布公告，2019年一季度实现营业收入同比增长，经营性现金流持续改善。",
            "光伏单晶硅片行业景气度回升，公司产能扩张顺利推进，盈利能力显著提升。",
            "公司半导体硅片业务进入放量阶段，8英寸和12英寸硅片需求持续增长。",
            "公司通过严格成本控制和精益化管理，有效降低经营成本，提升盈利能力。",
            "我们预计公司未来三年净利润将保持高速增长，维持对公司的增持评级。",
            "公司与地方政府签署合作协议，拟投资建设新的单晶硅项目，进一步扩大产能。",
            "随着光伏政策落地和海外需求旺盛，公司硅片价格企稳回升，盈利拐点明确。",
            "公司半导体材料业务毛利率提升，受益于产品结构升级和大尺寸硅片放量。",
            "公司发布员工持股计划，有助于绑定核心员工利益，提升公司治理水平。",
            "公司预计未来将迎来光伏与半导体双轮驱动的增长新时代。"
        ]

        # Negative / noise examples (table headers, metadata, disclaimers, etc.)
        self.neg_examples = [
            "主要股东（2019Q1）",
            "基本数据（截至2019年06月20日）",
            "收入结构（2018A）",
            "资产负债表",
            "现金流量表",
            "预测财务报表（单位：百万元）",
            "联系人：陈瑶",
            "当前股价：8.24元",
            "报告日期：2017年11月29日",
            "行业：电气设备和新能源",
            "总市值（亿元）",
            "流通A股（亿股）",
            "52周内股价区间（元）",
            "近3月换手率",
            "股价表现（最近一年）",
            "股价历史走势",
            "相对指数表现",
            "5.2/11.96",
            "27.55%",
            "0.39",
            "10.86亿",
            "单位：百万元",
            "指标",
            "投资评级体系与评级定义",
            "市场有风险，投资需谨慎",
            "免责条款",
            "公司基本情况（最新）",
            "《【联讯电新公司点评】中环股",
            "份(002129): 非公开发行助",
            "投资评级的说明:",
            "买入: 预期未来 6-12 个月内上涨幅度在 15%以上;",
            "增持: 预期未来 6-12 个月内上涨幅度在 5%-15%;",
            "中性: 预期未来 6-12 个月内变动幅度在 -5%-5%;",
            "减持: 预期未来 6-12 个月内下跌幅度在 5%以上。"

        ]


        # Precompute embeddings for example lists to speed up classification
        self.pos_emb = self.model.encode(self.pos_examples, convert_to_tensor=True)
        self.neg_emb = self.model.encode(self.neg_examples, convert_to_tensor=True)

    # ============================================================
    # classify: short-text heuristics added
    # ============================================================

    def classify(self, line: str) -> str:
        """Classify a single line as 'text' or 'noise'.

        The method uses embedding similarity against positive/negative example
        pools, plus a set of heuristics tailored for short lines (numeric-only
        lines, symbol-heavy lines, and a length-based penalty).
        """

        text = line.strip()
        if not text:
            return "noise"

        length = len(text)

        # -------------------------
        # 1. embedding similarity
        # -------------------------
        emb = self.model.encode(text, convert_to_tensor=True)
        pos_sim = util.cos_sim(emb, self.pos_emb).max().item()
        neg_sim = util.cos_sim(emb, self.neg_emb).max().item()

        # -------------------------
        # 2. Strong rule: prefer noise for very short lines that look numeric
        # -------------------------
        if length <= 20:
            if re.match(r"^[\d\.\-/%]+$", text) or \
            re.search(r"(亿|万|元|%|同比|环比|YOY|《|》|【|】)", text):
                neg_sim += 0.2  # heuristic penalty, tunable

        # -------------------------
        # 3. Short-text penalty: shorter lines receive larger penalty
        # -------------------------
        if length < 20:
            # penalty formula example: (20 - length) * 0.01
            neg_sim += (20 - length) * 0.01

            # # reduce penalty if short line contains clear body-text keywords
            # if re.search(r"(收入|净利润|增长|业务|产能)", text):
            #     neg_sim -= 0.2  # optional reduction, tunable

        # -------------------------
        # 4. Digit/symbol ratio penalty
        # -------------------------
        num_symbol_chars = len(re.findall(r"[\d\.\-/%]【】《》<>", text))
        ratio = num_symbol_chars / max(1, length)
        neg_sim += ratio * 0.3  # ratio penalty, tunable

        # -------------------------
        # 5. Final decision
        # -------------------------
        NEG_THRESHOLD = 0.65

        if neg_sim > NEG_THRESHOLD and neg_sim > pos_sim:
            return "noise"
        else:
            return "text"

# ============================================================
# 2. Recheck: context-aware secondary judgment
# ============================================================
clf = EmbeddingClassifier()

def recheck_with_context(lines, labels, clf, i):
    """Re-evaluate a possibly-noise line using surrounding context.

    This function constructs short candidates by joining the current line
    with neighboring lines (previous/next) or using the nearest labeled
    'text' lines and asks the classifier to re-evaluate. It is used to
    reduce false positives for short lines that only make sense together
    with context.
    """

    line = lines[i].strip()

    # Treat very short lines as noise by default
    if len(line.strip()) < 10:
        return "noise"

    if not line:
        return "noise"

    prev1 = lines[i-1] if i > 0 else ""

    # next1 = lines[i+1] if i < len(lines)-1 else ""

    # prev_pos = ""
    # for j in range(i-1, -1, -1):
    #     if labels[j] == "text":
    #         prev_pos = lines[j]
    #         break

    # next_pos = ""
    # for j in range(i+1, len(lines)):
    #     if labels[j] == "text":
    #         next_pos = lines[j]
    #         break

    candidates = [
        line,
        prev1 + "" + line,
        # line + "" + next1
        # prev_pos + " " + line + " " + next_pos,
    ]

    for text in candidates:
        if clf.classify(text) == "text":
            return "text"

    return "noise"


def batch_clean(input_dir, output_dir):
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
            clean_file(in_path, out_path)

def clean_file(input_path, output_path):
    """Label a single file: classify each line and optionally re-check using context.

    Output format: each output line is `label\t<original line>` where label
    is either `text` or `noise`.
    """
    print(f"Processing file: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()


    # First-pass classification using embeddings and heuristics
    labels = [clf.classify(line) for line in lines]

    # Second-pass: recheck lines labeled as 'noise' using neighboring context
    for i in range(len(lines)):
        if labels[i] == "noise":
            labels[i] = recheck_with_context(lines, labels, clf, i)

    print(f"Writing output to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as out_f:
        for label, line in zip(labels, lines):
            out_f.write(f"{label}\t{line}\n")

    print("File cleaned!")
# ============================================================
# 3. Main program
# ============================================================

def main():
    """Entry point: label all cleaned report files into a labeled directory."""
    input_path = r"reports_txt_by_quarter_cleaned"
    output_path = r"reports_txt_by_quarter_cleaned_labeled"
    batch_clean(input_path, output_path)
    print("labeling done!")


if __name__ == "__main__":
    main()
