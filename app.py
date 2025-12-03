# app.py

import streamlit as st
import pandas as pd
from io import BytesIO, StringIO
from bs4 import BeautifulSoup
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import requests
import json

from utils import pdf_to_image_buffers
from extractor_utils import (
    extract_all_data,
    extract_from_image_chunks_parallel
)


st.title("Bank Statement Extractor")

st.write("Upload a PDF and extract structured transaction data.")

uploaded_file = st.file_uploader("Upload a bank statement PDF", type=["pdf"])

if uploaded_file:
    pdf_filename = uploaded_file.name.replace(".pdf", "")
    st.success("PDF uploaded successfully!")

    if st.button("Start Extract"):
        with st.spinner("Converting PDF pages to images..."):
            image_buffers = pdf_to_image_buffers(uploaded_file)

        st.info(f"{len(image_buffers)} pages converted.")

        # API key stored in Streamlit Secrets
        api_key = st.secrets["NANONETS_API_KEY"]

        # Extract fields using first page markdown
        st.write("Extracting fields from first page...")

        # Send first page only
        import requests, json
        url = "https://extraction-api.nanonets.com/extract"
        headers = {"Authorization": f"Bearer {api_key}"}

        fp_files = {"file": (f"{pdf_filename}_first_page.png", image_buffers[0], "image/png")}
        first_page_data = {"output_type": "markdown-financial-docs", "model": "nanonets-ocr-s"}

        fp_resp = requests.post(url, files=fp_files, data=first_page_data, headers=headers).json()
        soup = BeautifulSoup(fp_resp["content"], "html.parser")
        tables = soup.find_all("table")

        dfs = [pd.read_html(StringIO(str(t)))[0] for t in tables]
        df_headers = max(dfs, key=lambda df: df.shape[1])
        fields = df_headers.columns.tolist()

        st.write("Fields detected:", ', '.join(fields))

        pages_data = {
            "output_type": "specified-fields",
            "model": "nanonets-ocr-s",
            "specified_fields": ", ".join(fields)
        }

        st.warning("Extracting all pages... This may take several minutes.")

        results = extract_from_image_chunks_parallel(
            image_buffers=image_buffers,
            pages_data=pages_data,
            api_key=api_key,
            pdf_filename=pdf_filename,
            max_workers=len(image_buffers)
        )

        st.success("Extraction completed!")

        pages_content = "\n---\n".join([r if isinstance(r, str) else "" for r in results])

        dfs = [extract_all_data(block, fields) for block in pages_content.split("---")]
        
        for i, df in enumerate(dfs):
            if isinstance(df, pd.DataFrame):
                df['page'] = i + 1
                df['row_number'] = range(1, len(df)+1)
        dfs = [d for d in dfs if isinstance(d, pd.DataFrame)]

        final_df = pd.concat(dfs).drop_duplicates().reset_index(drop=True)

        st.dataframe(final_df)

        csv = final_df.to_csv(index=False).encode()
        st.download_button("Download CSV", csv, f"{pdf_filename}.csv", "text/csv")
