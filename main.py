import os
from datetime import datetime
import argparse
from src.config import file_paths
from src.ui import launch_gradio_ui
from src.llm.ollamaModels import llm, embedder
from src.langchain_utils.qa_chain import get_qa_chain
from src.qdrant_utils.query import insert_into_db, get_retriever
from src.langchain_utils.document_handler import get_text_from_document
from src.qdrant_utils.connection import qdrant_connection


def main():
    def get_bot_response(message, history):
        # Handle greetings separately
        greetings = ["hello", "hi", "hey", "greetings", "namaste"]
        if any(greeting in message.lower() for greeting in greetings):
            return "Hello! I'm here to assist you with questions. Please feel free to ask anything."
        
        # Build the context from the history
        context = "\n".join(f"User: {msg}\nBot: {resp}" for msg, resp in history)
        context += f"\nUser: {message}\nBot:"

        try:
            result = qa_chain.invoke({"query": message, "context": context})
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
            file_contents += get_text_from_document(file_path)
            os.replace(file_path, f"{file_paths.archive}{file}")
        if file_contents:
            # insert_into_db(embedder, file_contents)
            conn.insert_data_to_qdrant(file_contents)
    else:
        print("Using existing collection without loading data")
    
    print(f"Start Retriving {datetime.now()}")
    a = conn.search_in_qdrant("just accounting")
    print(f"Retriving Completed {datetime.now()}")

    formatted_prompt = """
You are an AI assistant specialized in the Critical perspective in accounting.
Do not make up or infer information
Remember to always base your answers on the {context} provided and address the specific {question} asked.
Be concise in your responses while ensuring accuracy and completeness.
""".format(context = "\n".join([x.payload['text'] for x in a]), question = "just accounting")
    
    # retriever = get_retriever(embedder)
    # qa_chain = get_qa_chain(llm, retriever)
    
    print(f"Invoke LLM model {datetime.now()}")
    response = llm.generate([formatted_prompt])
    print(f"LLM invoke completed {datetime.now()}")
    
    print(response)
    
    # launch_gradio_ui(respond)
    

if __name__ == "__main__":
    main()