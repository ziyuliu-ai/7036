# Eastmoney Research Report Downloader

## Project Overview
This is a Python tool designed to automatically download research report PDFs from Eastmoney. Its main function is to batch‑download research reports for a specified stock, with support for intelligent classification, page‑count validation, and retry mechanisms.

## Key Features
- 🔍 **Smart Filtering:** Only downloads research reports with page counts exceeding a specified threshold
- 📁 **Automatic Categorization:** Creates separate directories for each stock code, with deep‑dive reports stored separately
- ✅ **Integrity Verification:** Automatically checks PDF page count after download to ensure completeness
- 🔄 **Retry Mechanism:** Automatically retries failed downloads up to 3 times
- 📅 **Time Range:** Supports custom query time ranges
- 🎯 **Smart Naming:** Automatically generates meaningful filenames based on report metadata


## Configuration File Description

The project uses a config.json configuration file with the following parameters:

```json
{
    "stock_code": "600519",
    "min_pages": 20,
    "download_dir": "reports_pdf",
    "years_ago": 2
}
```

### Configuration Parameter Details

| Parameter	| Type | Default | Description |
|------|------|--------|------|
| `stock_code` |	string | "600519" |	Stock code, e.g., 600519 for Kweichow Moutai |
| `min_pages`	| int	| 20 | Minimum page threshold; only reports exceeding this value are downloaded |
| `download_dir`	| string | "reports_pdf" |	Directory name for downloaded files |
| `years_ago`	| int	| 2	| How many years back to query reports |

## Install Dependency

```bash
pip install requests pycurl PyPDF2
```

### Dependency Description
- `requests`: HTTP request library
- `pycurl`: High‑performance download library
- `PyPDF2`: PDF file processing library


### Usage
1. **Configure Parameters:** Edit the `config.json` file to set the target stock code and other parameters
2. **Run the Program:** Execute the following command to start downloading

```bash
python main.py
```

## File Structure
After downloading, the file structure looks like this:

```
reports_pdf/
├── STOCKNAME/
│   ├── 20240526_BROKERNAME_REPORTTITLE.pdf
│   ├── DEPTHREPORT/
│   │   └── 20240526_BROKERNAME_REPORTTITLE.pdf
│   └── ...
├── raw_data/
│   ├── page_1_600519_2022-06-24_2024-06-24.json
│   ├── page_2_600519_2022-06-24_2024-06-24.json
│   └── ...
└── detail_data/
    ├── detail_AP202408091639215987.html
    ├── zwinfo_AP202408091639215987.json
    ├── detail_AP202408091639215331.html
    ├── zwinfo_AP202408091639215331.json
    └── ...
```

## Raw Data Storage
The program automatically saves the following raw data for debugging and later analysis:

### List Page Data (`raw_data/`)
- **Filename Format:** `page_{page}_{stock_code}_{start_date}_{end_date}.json`
- **Content:** Full API response of the research report list
- **Purpose:** Data backup, resume capability, offline analysis

### Detail Page Data (`detail_data/`)
- **HTML Files:** `detail_{reportID}.html`
- Full HTML source of the detail page
- Useful for debugging page structure issues

- **JSON Files:** `zwinfo_{reportID}.json`
- Parsed zwinfo data
- Contains PDF download link and naming information
- Useful for verifying filenames and download URLs


### Advantages of Data Storage
- **Completeness:** Preserves the full data pipeline from list to detail to PDF
- **Debugging Convenience:** Helps diagnose issues by reviewing raw data
- **Avoids Duplicate Requests:** Improves program efficiency
- **Further Analysis:** Enables deeper analysis based on saved data

## Feature Details
### Smart Naming Rules
PDF filename format: `Date_Organization_StockName_ReportTitle.pdf`

- Automatically removes redundant information
- Replaces special characters with underscores
- Sorted chronologically

### Deep Report Classification
- Reports exceeding the page threshold are automatically moved to the “Deep Reports” subdirectory
- Helps distinguish regular reports from in‑depth analyses

### Download Verification
- Automatically checks PDF page count after download
- Retries download if page count mismatches
- Up to 3 retries to ensure file integrity

## Notes
- **Network Environment:** Ensure a stable network connection
- **Storage Space:** PDF files may take significant space; ensure sufficient disk capacity
- **Access Frequency:** Built‑in delay mechanism prevents excessive server load
- **File Permissions:** Ensure the program has permission to create directories and files

