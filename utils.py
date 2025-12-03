import fitz  # PyMuPDF
from io import BytesIO

def pdf_to_image_buffers(uploaded_file):
    """
    Converts uploaded PDF into a list of image buffers (PNG).
    Works on Streamlit Cloud because it uses PyMuPDF (no Poppler).
    """

    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    image_buffers = []

    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        pix = page.get_pixmap(dpi=200)

        buf = BytesIO()
        buf.write(pix.tobytes("png"))
        buf.seek(0)

        image_buffers.append(buf)

    return image_buffers
