# extractor_utils.py

import re
import ast
import json
import time
import pandas as pd
import requests
from io import BytesIO, StringIO
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed


################## Key Mapping ##################

def build_key_map(fields):
    key_map = {}
    for field in fields:
        clean = field.strip()
        lower = clean.lower()
        snake = lower.replace(" ", "_")
        nospace = lower.replace(" ", "")

        key_map[lower] = clean
        key_map[snake] = clean
        key_map[nospace] = clean
    return key_map


def replace_keywords_in_string(content, key_map):
    if not isinstance(content, str):
        return content
    result = content
    for key, value in key_map.items():
        result = result.replace(key, value)
    return result


############## Extractors ##############

def quick_clean_block(content):
    transactions = []
    lines = content.split('\n')
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('{') and stripped.endswith('}'):
            try:
                transaction = ast.literal_eval(stripped)
                transactions.append(transaction)
            except:
                pass
    return pd.DataFrame(transactions)


def quick_clean_json(content):
    text = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
    blocks = re.findall(r'(?:json\s*)?(\[.*?\])', text, re.DOTALL)
    all_txns = []
    for b in blocks:
        try:
            all_txns.extend(json.loads(b))
        except:
            pass
    return pd.DataFrame(all_txns)


def extract_html_tables(content):
    soup = BeautifulSoup(content, "html.parser")
    tables = soup.find_all("table")
    df_list = [pd.read_html(StringIO(str(table)))[0] for table in tables]
    return pd.concat(df_list) if len(df_list) > 1 else df_list[0]


def extract_json_only(content):
    transactions = json.loads(content)
    if isinstance(transactions['content'], list):
        return pd.DataFrame(transactions["content"])
    elif isinstance(transactions['content'], dict):
        return pd.DataFrame.from_dict(transactions["content"], orient='index').T
    else:
        return pd.DataFrame()


def extract_all_data(content, fields):
    new_content = replace_keywords_in_string(content, build_key_map(fields))
    data = []

    if '<table' in new_content:
        data.append(extract_html_tables(new_content))

    if '[' in new_content and '{' in new_content and '}' in new_content and ']' in new_content and '"metadata"' not in new_content:
        data.append(quick_clean_json(new_content))

    if '{"' in new_content and '"}' in new_content and '"metadata"' not in new_content:
        data.append(quick_clean_block(new_content))

    if '"metadata' in new_content and '"content' in new_content:
        data.append(extract_json_only(new_content))

    return pd.concat(data) if len(data) > 0 else []


############## ASYNC IMAGE CHUNK EXTRACTION #################

def extract_from_image_chunks_parallel(
    image_buffers: List[BytesIO],
    pages_data: dict,
    api_key: str,
    pdf_filename: str,
    max_workers: int = 10
) -> List[Dict[Any, Any]]:

    url = "https://extraction-api.nanonets.com/extract-async"
    headers = {"Authorization": f"Bearer {api_key}"}

    def process_chunk(args):
        idx, buffer = args
        buffer.seek(0)

        custom_chunk_name = f"{pdf_filename}_page_{idx}.png"

        files = {"file": (custom_chunk_name, buffer, "image/png")}

        try:
            submit = requests.post(url, headers=headers, files=files, data=pages_data, timeout=60)
            submit.raise_for_status()
            record_id = submit.json()["record_id"]

            poll_url = f"https://extraction-api.nanonets.com/files/{record_id}"

            while True:
                r = requests.get(poll_url, headers=headers).json()
                if r.get("processing_status") == "completed":
                    return r.get("content", "")
                elif r.get("processing_status") in ["failed", "error"]:
                    return {"error": True, "chunk": idx, "details": r}
                time.sleep(5)

        except Exception as e:
            return {"error": True, "chunk": idx, "details": str(e)}

    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = [
            exe.submit(process_chunk, (i + 1, buf))
            for i, buf in enumerate(image_buffers)
        ]

    results = [f.result() for f in as_completed(futures)]
    return results
