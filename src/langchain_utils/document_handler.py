import re
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

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s,.]', '', text)
    return text.strip()

def chunk_text(text, chunk_size, overlap):
    chunks = []
    
    # # Identify the Part to be used as Metadata
    pattern = r'Part-\d+ \n[^\n]+ \n'
    matches = re.finditer(pattern, text)    
    indexes_in_part = []
    parts = []
    
    for match in matches:
        start_index = match.start()
        end_index = match.end()
        indexes_in_part.append((start_index, end_index))
        
    # # CREATE_CHUNKS
    number_of_parts = len(indexes_in_part)
    for i in range(number_of_parts):
        chunks = []
        title = clean_text(text[indexes_in_part[i][0]:indexes_in_part[i][1]])

        for i in range(indexes_in_part[i][1], indexes_in_part[i+1][0] if number_of_parts -1 != i else len(text), chunk_size - overlap):
            chunk = clean_text(text[i:i + chunk_size])
            chunks.append(chunk)
        parts.append({
            "title": title,
            "chunks": chunks
        })
        
    return parts


def get_text_from_document(file_path, chunk_size = 1000, chunk_overlap = 100):
    if ".pdf" in file_path:
        text = extract_text_from_pdf(file_path)
    else:
        text = extract_text_from_text_file(file_path)
    
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    return chunks
    