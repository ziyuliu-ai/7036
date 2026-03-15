"""
Generate day-level NLP dataset from reports_txt_by_day.

Input:
- report text in reports_txt_by_day/YYYYMMDD/公司名/*.txt

Label:
- company stock cumulative return from report date over next N months
  (default N=3, compounded from daily returns in all_stock_returns.csv)

Output:
- a regression dataset CSV for BERT training
- optional rolling-window splits for time-series training
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class WindowSplit:
    split_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    train_df: pd.DataFrame
    val_df: pd.DataFrame


class DayLevelDatasetBuilder:
    def __init__(
        self,
        reports_dir: str = "reports_txt_by_day",
        returns_file: str = "all_stock_returns.csv",
        mapping_file: str = "Eastmoney_report_pdf_download/HS300.csv",
        min_text_length: int = 50,
        horizon_months: int = 3,
    ) -> None:
        self.reports_dir = Path(reports_dir)
        self.returns_file = Path(returns_file)
        self.mapping_file = Path(mapping_file)
        self.min_text_length = min_text_length
        self.horizon_months = horizon_months

        if not self.returns_file.exists():
            alt = Path("all_stock_return.csv")
            if alt.exists():
                self.returns_file = alt

        self.name_to_code = self._load_name_to_code()
        self.returns_df = self._load_returns()

    def _load_name_to_code(self) -> Dict[str, str]:
        mapping = pd.read_csv(self.mapping_file, dtype={"股票代码": str})
        mapping["股票代码"] = mapping["股票代码"].str.zfill(6)
        return dict(zip(mapping["股票简称"], mapping["股票代码"]))

    def _load_returns(self) -> pd.DataFrame:
        df = pd.read_csv(self.returns_file)
        df["交易日期"] = pd.to_datetime(df["交易日期"], errors="coerce")
        df = df.dropna(subset=["交易日期"]).sort_values("交易日期")

        for col in df.columns:
            if col == "交易日期":
                continue
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    @staticmethod
    def _read_text(txt_path: Path) -> Optional[str]:
        for enc in ("utf-8", "gbk", "gb18030"):
            try:
                text = txt_path.read_text(encoding=enc).strip()
                return text
            except Exception:
                continue
        return None

    def _resolve_stock_code(self, company_name: str) -> Optional[str]:
        if company_name in self.name_to_code:
            return self.name_to_code[company_name]

        if company_name.isdigit():
            code = company_name.zfill(6)
            if code in self.returns_df.columns:
                return code

        return None

    def _calc_forward_return(self, stock_code: str, report_date: pd.Timestamp) -> Optional[float]:
        if stock_code not in self.returns_df.columns:
            return None

        end_date = report_date + pd.DateOffset(months=self.horizon_months)
        mask = (self.returns_df["交易日期"] >= report_date) & (self.returns_df["交易日期"] <= end_date)
        series = self.returns_df.loc[mask, stock_code].dropna()

        if series.empty:
            return None

        daily_return = series / 100.0
        cumulative_return = (1.0 + daily_return).prod() - 1.0
        return float(cumulative_return)

    def build_dataset(self) -> pd.DataFrame:
        rows: List[Dict] = []

        if not self.reports_dir.exists():
            raise FileNotFoundError(f"reports_dir not found: {self.reports_dir}")

        for day_dir in sorted(self.reports_dir.iterdir()):
            if not day_dir.is_dir() or not day_dir.name.isdigit() or len(day_dir.name) != 8:
                continue

            report_date = pd.to_datetime(day_dir.name, format="%Y%m%d", errors="coerce")
            if pd.isna(report_date):
                continue

            for company_dir in day_dir.iterdir():
                if not company_dir.is_dir():
                    continue

                company_name = company_dir.name
                stock_code = self._resolve_stock_code(company_name)
                if stock_code is None:
                    continue

                label = self._calc_forward_return(stock_code, report_date)
                if label is None:
                    continue

                for txt_path in company_dir.rglob("*.txt"):
                    text = self._read_text(txt_path)
                    if not text or len(text) < self.min_text_length:
                        continue

                    rows.append(
                        {
                            "text": text,
                            "label": label,
                            "report_date": report_date.strftime("%Y-%m-%d"),
                            "stock_code": stock_code,
                            "company_name": company_name,
                            "horizon_months": self.horizon_months,
                            "source_file": str(txt_path.as_posix()),
                        }
                    )

        dataset = pd.DataFrame(rows)
        if dataset.empty:
            return dataset

        dataset["report_date"] = pd.to_datetime(dataset["report_date"])
        dataset = dataset.sort_values(["report_date", "stock_code"]).reset_index(drop=True)
        return dataset


def build_rolling_windows(
    dataset: pd.DataFrame,
    train_months: int = 24,
    val_months: int = 3,
    step_months: int = 1,
) -> List[WindowSplit]:
    if dataset.empty:
        return []

    df = dataset.copy()
    df["report_date"] = pd.to_datetime(df["report_date"])
    df = df.sort_values("report_date")

    min_date = df["report_date"].min().normalize()
    max_date = df["report_date"].max().normalize()

    splits: List[WindowSplit] = []
    split_id = 0

    val_start = min_date + pd.DateOffset(months=train_months)
    while val_start < max_date:
        train_start = val_start - pd.DateOffset(months=train_months)
        train_end = val_start
        val_end = val_start + pd.DateOffset(months=val_months)

        train_mask = (df["report_date"] >= train_start) & (df["report_date"] < train_end)
        val_mask = (df["report_date"] >= val_start) & (df["report_date"] < val_end)

        train_df = df.loc[train_mask].reset_index(drop=True)
        val_df = df.loc[val_mask].reset_index(drop=True)

        if not train_df.empty and not val_df.empty:
            split_id += 1
            splits.append(
                WindowSplit(
                    split_id=split_id,
                    train_start=train_start,
                    train_end=train_end,
                    val_start=val_start,
                    val_end=val_end,
                    train_df=train_df,
                    val_df=val_df,
                )
            )

        val_start = val_start + pd.DateOffset(months=step_months)

    return splits


def save_rolling_windows(splits: List[WindowSplit], output_dir: str = "rolling_windows") -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_rows: List[Dict] = []
    for split in splits:
        train_path = out_dir / f"train_split_{split.split_id:03d}.csv"
        val_path = out_dir / f"val_split_{split.split_id:03d}.csv"

        split.train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
        split.val_df.to_csv(val_path, index=False, encoding="utf-8-sig")

        meta_rows.append(
            {
                "split_id": split.split_id,
                "train_start": split.train_start.strftime("%Y-%m-%d"),
                "train_end_exclusive": split.train_end.strftime("%Y-%m-%d"),
                "val_start": split.val_start.strftime("%Y-%m-%d"),
                "val_end_exclusive": split.val_end.strftime("%Y-%m-%d"),
                "train_size": len(split.train_df),
                "val_size": len(split.val_df),
                "train_file": train_path.as_posix(),
                "val_file": val_path.as_posix(),
            }
        )

    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_csv(out_dir / "rolling_splits_meta.csv", index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build day-level text-return dataset with rolling splits")
    parser.add_argument("--reports-dir", default="reports_txt_by_day")
    parser.add_argument("--returns-file", default="all_stock_returns.csv")
    parser.add_argument("--mapping-file", default="Eastmoney_report_pdf_download/HS300.csv")
    parser.add_argument("--output-file", default="daily_text_3m_return_dataset.csv")
    parser.add_argument("--horizon-months", type=int, default=3)
    parser.add_argument("--min-text-length", type=int, default=50)

    parser.add_argument("--build-rolling", action="store_true")
    parser.add_argument("--train-months", type=int, default=24)
    parser.add_argument("--val-months", type=int, default=3)
    parser.add_argument("--step-months", type=int, default=1)
    parser.add_argument("--rolling-output-dir", default="rolling_windows")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    builder = DayLevelDatasetBuilder(
        reports_dir=args.reports_dir,
        returns_file=args.returns_file,
        mapping_file=args.mapping_file,
        min_text_length=args.min_text_length,
        horizon_months=args.horizon_months,
    )

    dataset = builder.build_dataset()
    if dataset.empty:
        print("No valid samples generated. Please check paths and data coverage.")
        return

    dataset.to_csv(args.output_file, index=False, encoding="utf-8-sig")
    print(f"Dataset saved: {args.output_file}, samples={len(dataset)}")
    print(
        "Date range: "
        f"{dataset['report_date'].min().strftime('%Y-%m-%d')} -> "
        f"{dataset['report_date'].max().strftime('%Y-%m-%d')}"
    )

    if args.build_rolling:
        splits = build_rolling_windows(
            dataset=dataset,
            train_months=args.train_months,
            val_months=args.val_months,
            step_months=args.step_months,
        )
        save_rolling_windows(splits, output_dir=args.rolling_output_dir)
        print(f"Rolling splits saved: {args.rolling_output_dir}, splits={len(splits)}")


if __name__ == "__main__":
    main()
