import fitz 

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(pdf_file)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    doc.close()
    return text

def extract_text_from_text_file(text_file):
    with open(text_file, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def chunk_text(text, chunk_size, overlap):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    return chunks


def get_text_from_document(file_path, chunk_size = 1000, chunk_overlap = 100):
    if ".pdf" in file_path:
        text = extract_text_from_pdf(file_path)
    else:
        text = extract_text_from_text_file(file_path)
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    return chunks
    