import os
import argparse
from tqdm import tqdm
from datetime import datetime
from src.config import file_paths
from src.questions import QUESTIONS
from src.ui import launch_gradio_ui
from src.llm.reranker import reranker
from src.llm.ollamaModels import check
from src.config import CUSTOM_PROMPT, ollama_configs, qdrant_configs
from src.llm.ollamaModels import llm, embedder
from src.qdrant_utils.connection import qdrant_connection
from src.langchain_utils.document_handler import get_text_from_document

def get_total_difference_seconds(start: datetime, end: datetime) -> int:
    return (end - start).total_seconds()

def multi_replace(value: str) -> str:
    return value.replace("\n", ":").replace(",", ";").replace('\u2705', "")

def perform_context_retrieval(conn: qdrant_connection, question: str, use_rerank: bool = True, return_retrieve_interval: bool = False):
    retrieve_start = datetime.now()
    retrieved_documents = conn.search_in_qdrant(question)
    contexts = [multi_replace(x.payload[ollama_configs.answer_key]) for x in retrieved_documents]
    retrieve_complete = datetime.now()
    
    
    contexts = conn.rerank_documents(question, contexts) if use_rerank else contexts
    rerank_complete = datetime.now()
    
    if return_retrieve_interval:
        database_retrieval_delta = get_total_difference_seconds(retrieve_start, retrieve_complete)
        rerank_delta = get_total_difference_seconds(retrieve_complete, rerank_complete)
        return contexts, database_retrieval_delta, rerank_delta
    else:
        return contexts
    

def main():
    def get_bot_response(message, history):
        contexts = perform_context_retrieval(conn, message)
        print(message)
        print(contexts)
        print("-"*40)
        if contexts:
            prompt = CUSTOM_PROMPT.format(context = "\n".join(contexts), question = message)
        else:
            return "Related documents not found"
        try:
            answer = llm.invoke(prompt)
            return answer
        except Exception as e:
            print(f"Error in get_bot_response: {str(e)}")
            return "I'm sorry, I encountered an error while processing your question."
    
    def respond(message, history = []):
        bot_response = get_bot_response(message, history) # # HISTORY NOT IN USE RIGHT NOW ... FASTER PROCESSING
        history.append((message, bot_response))
        return "", history
    
    def adjust_list_length(lst, length = qdrant_configs.K, padding_value='-'):
        if len(lst) > length:
            return lst[:length]  # Trim the list
        else:
            return lst + [padding_value] * (length - len(lst))  # Pad the list
    
    conn = qdrant_connection(embedder, reranker)
    conn.create_collection()
    
    parser = argparse.ArgumentParser(description = 'Arguments')
    parser.add_argument('--load', help = 'Choose whether to load data to qdrant or not', default = False)
    parser.add_argument('--ui', help = 'Choose whether to prepare UI or answer existing question', default = False)
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

    if args.ui:
        launch_gradio_ui(respond)
    else:
        output_file = f"{file_paths.output_file_path}{file_paths.output_file_name}"
        with open(output_file, 'w') as file:
            titles = [f"Retrieved Document {i + 1}" for i in range(qdrant_configs.K)]
            file.write(f"Question, Database Retrieval Delta, Invoke Model Delta, Rerank Delta, {', '.join(titles)}, Answer, Generated Answer, Check Answer\n")
        # print("Question, Database Retrieval Delta, Invoke Model Delta, Retrieved Context, Answer, Generated Answer\n")
        for i in tqdm(QUESTIONS):
        # for i in QUESTIONS: # tqdm(QUESTIONS):
            answer_check = "No"
            question = i["question"]
            answer = i["answer"]

            # # SEARCH IN DATABASE      
            contexts, database_retrieval_delta, rerank_delta = perform_context_retrieval(conn, question, return_retrieve_interval=True)

            if not contexts:
                with open(output_file, 'a') as file:
                    file.write(f"{multi_replace(question)}, , , , , {multi_replace(answer)}, 'Related documents not found'\n")
                continue
            context_str = "\n".join(contexts)
            prompt = CUSTOM_PROMPT.format(context = context_str, question = question)

            # # INVOKE MODEL
            invoke_start = datetime.now()
            generated_answer = llm.invoke(prompt)
            invoke_complete = datetime.now()
            invoke_model_delta = get_total_difference_seconds(invoke_start, invoke_complete)

            # # CHECK ANSWER AVAILABILITY IN CONTEXT
            answer_check = check(context_str, generated_answer)
            
            # WRITE TO FILE
            with open(output_file, 'a') as file:
                contexts = adjust_list_length(contexts)

                generated_answer = generated_answer.replace("\n", ":").replace(",", ";")
                question = question.replace("\n", ":").replace(",", ";")
                answer = answer.replace("\n", ":").replace(",", ";")
                
                # print(f"{multi_replace(question)}, {database_retrieval_delta}, {invoke_model_delta}, {multi_replace(retrieved_context)}, {multi_replace(answer)}, {multi_replace(generated_answer)}\n")
                file.write(f"{multi_replace(question)}, {database_retrieval_delta}, {invoke_model_delta}, {rerank_delta}, {', '.join(contexts)}, {multi_replace(answer)}, ANSWER: {multi_replace(generated_answer)}, {answer_check}\n")
    
if __name__ == "__main__":
    main()