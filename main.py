import os
import argparse
from tqdm import tqdm
from datetime import datetime
from src.config import file_paths
from src.questions import QUESTIONS
from src.llm.ollamaModels import llm, embedder
from src.config import CUSTOM_PROMPT, ollama_configs
from src.qdrant_utils.connection import qdrant_connection
from src.langchain_utils.document_handler import get_text_from_document
from src.qdrant_utils.connection import qdrant_connection

def get_total_difference_seconds(start: datetime, end: datetime):
    return (end - start).total_seconds()

def main():
    conn = qdrant_connection(embedder)
    conn.create_collection()
    
    parser = argparse.ArgumentParser(description = 'Arguments')
    parser.add_argument('--load', help = 'Choose whether to load data to qdrant or not', default = False)
    args = parser.parse_args()
    
    if args.load:
        
        print("Loading new data from files")
        file_contents = []
        files_list = os.listdir(file_paths.new)
        
        for file in files_list:
            file_path = f"{file_paths.new}{file}"
            file_contents += get_text_from_document(file_path, chunk_size = 2048, chunk_overlap = 128)
            # os.replace(file_path, f"{file_paths.archive}{file}")
        if file_contents:
            conn.insert_data_to_qdrant(file_contents)
    else:
        print("Using existing collection without loading data")

    output_file = f"{file_paths.output_file_path}{file_paths.output_file_name}"
    with open(output_file, 'w') as file:
        file.write("Question, Database Retrieval Delta, Invoke Model Delta, Answer, Generated Answer\n")
    for i in tqdm(QUESTIONS):
        question = i["question"]
        answer = i["answer"]

        # # SEARCH IN DATABASE
        retrieve_start = datetime.now()
        context = conn.search_in_qdrant(question)
        retrieve_complete = datetime.now()
        
        prompt = CUSTOM_PROMPT.format(context = "\n".join([x.payload[ollama_configs.answer_key] for x in context]), question = question)

        # # INVOKE MODEL
        invoke_start = datetime.now()
        generated_answer = llm.invoke(prompt)
        invoke_complete = datetime.now()
        
        # WRITE TO FILE
        with open(output_file, 'a') as file:
            database_retrieval_delta = get_total_difference_seconds(retrieve_start, retrieve_complete)
            invoke_model_delta = get_total_difference_seconds(invoke_start, invoke_complete)
            generated_answer.replace("\n", " ").replace(",", ";")
            question.replace("\n", " ").replace(",", ";")
            answer.replace("\n", " ").replace(",", ";")
            
            file.write(f"{question}, {database_retrieval_delta}, {invoke_model_delta}, {answer}, {generated_answer}\n")
    

if __name__ == "__main__":
    main()