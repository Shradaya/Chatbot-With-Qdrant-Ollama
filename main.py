import os
from datetime import datetime
import argparse
from src.config import file_paths
from src.questions import QUESTIONS
from src.llm.ollamaModels import llm, embedder
from src.config import CUSTOM_PROMPT, ollama_configs
from src.qdrant_utils.connection import qdrant_connection
from src.langchain_utils.document_handler import get_text_from_document
from src.qdrant_utils.connection import qdrant_connection


def main():
    def get_bot_response(message, history):
        try:
            print(message)
            result = qa_chain.invoke({"query": message})
            answer = result['result']
            return answer
        except Exception as e:
            print(f"Error in get_bot_response: {str(e)}")
            return "I'm sorry, I encountered an error while processing your question."

    
    def respond(message, history = []):
        bot_response = get_bot_response(message, history)
        history.append((message, bot_response))
        return "", history
    
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
    
    easy_questions = QUESTIONS['Easy']
    for i in easy_questions:
        question = i["question"]
        
        # # SEARCH IN DATABASE
        print(f"Start Retriving {datetime.now()}")
        context = conn.search_in_qdrant(question)
        print(f"Retriving Completed {datetime.now()}")
        prompt = CUSTOM_PROMPT.format(context = "\n".join([x.payload[ollama_configs.answer_key] for x in context]), question = question)

        print(f"Start Invoking {datetime.now()}")
        print(llm.invoke(prompt))
        print(f"Completed {datetime.now()}")
        break
    

if __name__ == "__main__":
    main()