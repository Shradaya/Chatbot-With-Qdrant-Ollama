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
    text = re.sub(r'\d+\.\s*', ' ', text)  
    text = re.sub(r'[^\w\s,:]', '', text)
    text = re.sub(r'\s+', ' ', text).strip() 
    text = re.sub(r'Part\d+\s*', '', text) 
    
    return text

def chunk_text(text, chunk_size, overlap):
    def divide_by_article(text):
        def remove_index(text):
            return re.sub(r'\b\d+\.\s*\n?|\s*\n', '', text)
        text = re.sub(r' \n \n\d+ \n \n', ' ', text)
        split_text = re.split(r' \n \n\d+. \n', text)

        # Remove any empty strings from the result
        split_text = [remove_index(text) for text in split_text if text and text != " \n"]
        return split_text
    
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
        if title == "Fundamental Rights and Duties":
            pass
        if i == number_of_parts - 1:
            start = indexes_in_part[i][1]
            splitted_texts = divide_by_article(text[start:])
        else:
            start = indexes_in_part[i][1]
            end = indexes_in_part[i+1][0]    
            splitted_texts = divide_by_article(text[start:end])
        for splitted_text in splitted_texts:
            cleaned = clean_text(splitted_text)
            if not cleaned:
                continue
            chunks.append(cleaned)
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
    