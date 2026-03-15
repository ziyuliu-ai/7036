import requests
import json
import re
import os
from urllib.parse import urljoin
from time import sleep
import pycurl
from io import BytesIO
from PyPDF2 import PdfReader
from datetime import datetime, timedelta
import pandas as pd
import time

def set_stock_code(code):
    # Set the global stock code used by API requests
    global STOCK_CODE
    STOCK_CODE = code

# Global configuration
BASE_URL = "https://reportapi.eastmoney.com/report/list"
DETAIL_BASE_URL = "https://data.eastmoney.com/report/info/"

MIN_PAGES = 2
DOWNLOAD_DIR = "reports_pdf"
YEARS_AGO = 10
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# List of User-Agent strings to rotate
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0"
]

def get_random_user_agent():
    """Return a random User-Agent string."""
    import random
    return random.choice(USER_AGENTS)

def fetch_jsonp_data(page_no=1):
    """
    Fetch research report listing data (JSONP -> JSON)
    :param page_no: Page number
    :return: Parsed data dictionary
    """
    # calculate date range
    today = datetime.today()
    end_time = today.strftime('%Y-%m-%d')
    begin_time = (today - timedelta(days=365*YEARS_AGO)).strftime('%Y-%m-%d')
    
    # Check whether a cached raw data file exists
    raw_data_dir = "raw_data"
    raw_data_file = os.path.join(raw_data_dir, f"page_{page_no}_{STOCK_CODE}_{begin_time}_{end_time}.json")
    
    if os.path.exists(raw_data_file):
        print(f"Using cached raw data: {raw_data_file}")
        try:
            with open(raw_data_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"read cached data failed: {e}")
    
    params = {
        "cb": "datatable6333112",
        "pageNo": page_no,
        "pageSize": 50,
        "code": STOCK_CODE,
        "industryCode": "*",
        "industry": "*",
        "rating": "*",
        "ratingchange": "*",
        "beginTime": begin_time,
        "endTime": end_time,
        "fields": "",
        "qType": 0,
        "p": page_no,
        "pageNum": page_no,
        "pageNumber": page_no,
        "_": int(time.time() * 1000)  # use current timestamp
    }
    headers = {
        "User-Agent": get_random_user_agent(),
        "Referer": "https://data.eastmoney.com/"
    }
    try:
        response = requests.get(BASE_URL, params=params, headers=headers)
        response.raise_for_status()
        # Extract JSON content from the JSONP response
        json_str = re.search(r'\((.*)\)', response.text).group(1)
        data = json.loads(json_str)
        
        # Save raw data to local cache
        if not os.path.exists(raw_data_dir):
            os.makedirs(raw_data_dir, exist_ok=True)
        
        with open(raw_data_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Raw data saved: {raw_data_file}")
        return data
    except Exception as e:
        print(f"get page {page_no} data failed: {e}")
        return None

def get_report_detail(info_code):
    """
    Fetch the report detail page HTML
    :param info_code: Report ID
    :return: Detail page HTML content
    """
    # Check whether the detail HTML has been cached
    detail_data_dir = "detail_data"
    detail_html_file = os.path.join(detail_data_dir, f"detail_{info_code}.html")
    
    if os.path.exists(detail_html_file):
        print(f"Using cached detail HTML: {detail_html_file}")
        try:
            with open(detail_html_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"read cached detail HTML failed: {e}")
    
    url = urljoin(DETAIL_BASE_URL, f"{info_code}.html")
    headers = {
        "User-Agent": get_random_user_agent(),
        "Referer": "https://data.eastmoney.com/"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Save fetched detail HTML to cache
        if not os.path.exists(detail_data_dir):
            os.makedirs(detail_data_dir, exist_ok=True)
        
        with open(detail_html_file, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print(f"Detail HTML saved: {detail_html_file}")
        return response.text
    except Exception as e:
        print(f"get report{info_code} failed: {e}")
        return None

def parse_detail_page(html, info_code):
    """
    Parse the detail page to extract the PDF download URL and metadata
    :param html: Detail page HTML
    :param info_code: Report ID
    :return: dict containing PDF URL and naming fields
    """
    try:
        # Extract the `zwinfo` JavaScript variable using regex
        match = re.search(r'var zwinfo\s*=\s*({.*?});', html, re.DOTALL)
        if not match:
            return None
        zwinfo = json.loads(match.group(1))
        
        # Save parsed `zwinfo` JSON for debugging
        detail_data_dir = "detail_data"
        zwinfo_file = os.path.join(detail_data_dir, f"zwinfo_{info_code}.json")
        with open(zwinfo_file, 'w', encoding='utf-8') as f:
            json.dump(zwinfo, f, ensure_ascii=False, indent=2)
        
        print(f"zwinfo data saved: {zwinfo_file}")
        
        # Extract required fields from zwinfo
        return {
            'attach_url': zwinfo.get('attach_url'),
            'notice_title': zwinfo.get('notice_title', ''),
            'short_name': zwinfo.get('short_name', ''),
            'notice_date': zwinfo.get('notice_date', ''),
            'source_sample_name': zwinfo.get('source_sample_name', ''),
            'attach_pages': zwinfo.get('attach_pages', '')
        }
    except Exception as e:
        print(f"parsing detail page failed: {e}")
        return None

def is_pdf_complete(pdf_path, expected_pages):
    """
    Check whether a downloaded PDF has the expected number of pages
    :param pdf_path: Path to the PDF file
    :param expected_pages: Expected page count (int)
    :return: (bool, actual_pages)
    """
    try:
        with open(pdf_path, 'rb') as f:
            reader = PdfReader(f)
            actual_pages = len(reader.pages)
        return actual_pages == expected_pages, actual_pages
    except Exception as e:
        print(f"Failed to read PDF pages: {e}")
        return False, 0

def download_pdf(pdf_url, filename):
    """
    Download a PDF using pycurl (imitates curl curl request)

    Parameters:
        pdf_url (str): URL of the PDF file
        filename (str): Filename to save (relative to DOWNLOAD_DIR)

    Returns:
        bool: True if download succeeded
    """
    save_path = os.path.join(DOWNLOAD_DIR, filename)
    buffer = BytesIO()
    c = pycurl.Curl()
    
    try:
        # Configure curl options
        c.setopt(pycurl.URL, pdf_url)
        c.setopt(pycurl.WRITEDATA, buffer)
        c.setopt(pycurl.FOLLOWLOCATION, True)
        c.setopt(pycurl.MAXREDIRS, 5)
        c.setopt(pycurl.CONNECTTIMEOUT, 30)
        c.setopt(pycurl.TIMEOUT, 300)
        
        # Set headers to help avoid trivial anti-scraping checks
        headers = [
            f"User-Agent: {get_random_user_agent()}",
            "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Referer: https://data.eastmoney.com/",
            "Accept-Language: zh-CN,zh;q=0.9"
        ]
        c.setopt(pycurl.HTTPHEADER, headers)
        
        # Perform the download
        c.perform()
        
        # Verify HTTP response code
        if c.getinfo(pycurl.HTTP_CODE) != 200:
            print(f"Download failed HTTP {c.getinfo(pycurl.HTTP_CODE)}")
            return False
            
        # Write buffer to file
        with open(save_path, 'wb') as f:
            f.write(buffer.getvalue())
            
        print(f"✓ Successfully downloaded {filename}")
        return True
        
    except pycurl.error as e:
        errno, errstr = e.args
        print(f"pycurl error ({errno}): {errstr}")
        return False
    except Exception as e:
        print(f"Download exception: {str(e)}")
        return False
    finally:
        c.close()
        buffer.close()

def process_all_reports():
    """Process all research reports for the currently set stock code."""
    # get first page
    first_page_data = fetch_jsonp_data(1)
    if not first_page_data:
        return
    
    total_page = first_page_data.get("TotalPage", 1)
    total_reports = first_page_data.get("hits", 0)
    print(f"Found {total_reports} research reports across {total_page} pages")
    
    # Iterate through all pages
    for page in range(1, total_page + 1):
        print(f"\nProcessing page {page}/{total_page}...")
        # get current page
        if page == 1:
            page_data = first_page_data
        else:
            page_data = fetch_jsonp_data(page)
            if not page_data:
                continue
        # handle each report
        for report in page_data.get("data", []):
            info_code = report.get("infoCode")
            if not info_code:
                continue
            
            # Check number of attached pages; skip if below MIN_PAGES
            attach_pages = report.get("attachPages", 0)
            try:
                attach_pages = int(attach_pages)
            except (ValueError, TypeError):
                attach_pages = 0
            
            if attach_pages < MIN_PAGES:
                print(f"Skipping report with insufficient pages: {report.get('title')} (pages: {attach_pages})")
                continue
                
            print(f"\nProcessing report: {report.get('title')} [{info_code}] (pages: {attach_pages})")
            # get html detail 
            detail_html = get_report_detail(info_code)
            if not detail_html:
                continue
            # parse pdf url and metadata
            detail_info = parse_detail_page(detail_html, info_code)
            if not detail_info or not detail_info.get('attach_url'):
                print("PDF link not found")
                continue
            # construct filename and path
            notice_title = detail_info.get('notice_title', '').strip().replace('/', '_')
            short_name = detail_info.get('short_name', '').strip().replace('/', '_')
            notice_date = detail_info.get('notice_date', '').replace('-', '')[:8]  # take YYYYMMDD only
            source_sample_name = detail_info.get('source_sample_name', '').strip().replace('/', '_')

            filename_parts = []
            filename_parts.append(notice_date)
            # check source_sample_name existence and uniqueness in filename
            if source_sample_name and source_sample_name not in notice_title:
                filename_parts.append(source_sample_name)
            # check short_name existence and uniqueness in filename
            if short_name and short_name not in notice_title:
                filename_parts.append(short_name)
            filename_parts.append(notice_title)
            # split filename and path
            pdf_filename = f"{'_'.join(filename_parts)}.pdf"
            pdf_subdir = f"{short_name}"
            
            # If it's an in-depth report (10+ pages), put it in a subfolder
            if attach_pages >= 10:
                pdf_subdir = f"{short_name}/DeepReports"
            
            pdf_full_path = os.path.join(DOWNLOAD_DIR, pdf_subdir, pdf_filename)
            
            # Ensure target directory exists
            pdf_dir = os.path.join(DOWNLOAD_DIR, pdf_subdir)
            if not os.path.exists(pdf_dir):
                os.makedirs(pdf_dir, exist_ok=True)
                print(f"Created directory: {pdf_dir}")
            
            # Skip if file already exists
            if os.path.exists(pdf_full_path):
                print(f"File exists, skipping download: {pdf_full_path}")
                continue
                
            # Download PDF and validate page count; retry up to `max_retries`
            max_retries = 3
            for attempt in range(1, max_retries + 1):
                download_pdf(detail_info['attach_url'], os.path.join(pdf_subdir, pdf_filename))
                # Validate PDF page count
                try:
                    expected_pages = int(detail_info.get('attach_pages', 0))
                except Exception:
                    expected_pages = 0
                is_complete = True
                actual_pages = 0
                if expected_pages > 0:
                    is_complete, actual_pages = is_pdf_complete(pdf_full_path, expected_pages)
                    if is_complete:
                        print(f"✓ PDF page count verified: {actual_pages} pages")
                        break
                    else:
                        print(f"✗ PDF page mismatch: actual {actual_pages}, expected {expected_pages}. Retrying ({attempt}/{max_retries})...")
                        # remove incomplete file before retrying
                        try:
                            os.remove(pdf_full_path)
                        except Exception:
                            pass
                        sleep(1)
                else:
                    break
            else:
                print(f"!!! PDF still incomplete after multiple attempts: {pdf_full_path}")
            # Polite delay between downloads
            sleep(1)



def main():
    start_time = time.time()

    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, "HS300.csv")

    df = pd.read_csv(csv_path, dtype=str)
    code_list = list(df['股票代码'])
    name_lst = list(df['股票简称'])
    stock_list = list(zip(code_list, name_lst))

    # 2. download reports for each stock
    for code, name in stock_list:
        print(f"\n==============================")
        print(f"start download {code} {name} reports")
        print(f"==============================")

        set_stock_code(code)
        process_all_reports()

        print(f"finish {code} {name} downloading\n")
        time.sleep(1)  # delay between stocks

    end_time = time.time()
    print(f"\n download finished in: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
