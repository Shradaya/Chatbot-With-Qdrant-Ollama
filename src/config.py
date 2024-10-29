class app:
    title = "Nepal's Constitution"

class ollama_configs:
    test = True
    re_ranker = "Available"
    # # Gemma2, 2B
    # model_name = "gemma2:2b"
    # vector_size = "2304"
    # answer_key = "text"
    
    # # # PHI 3, 3.8B
    # model_name = "phi3:3.8B" # 3.8 B
    # vector_size = "3072"
    # answer_key = "text"
    
    # # # llama 3.1, 8B
    model_name = "llama3.1"
    vector_size = "4096"
    answer_key = "text"
    
class file_paths:
    new = "./pdfs/"
    archive = "./archived_pdfs/"
    output_file_path = "./answers_collection/"
    output_file_name = f"0.4-DIVIDE-{ollama_configs.model_name.replace('.', '-').replace(':', '-')}{'-TEST' if ollama_configs.test else ''}.csv"

class qdrant_configs:
    K = 3
    RETRIEVE_COUNT = 15
    HOST = "localhost"
    PORT = "6333"
    URI = f"http://{HOST}:{PORT}/"
    COLLECTION = f"constitution-{ollama_configs.model_name.replace('.', '-').replace(':', '-')}{'-TEST' if ollama_configs.test else ''}"
    VECTOR_SIZE = ollama_configs.vector_size
    DISTANCE = "Cosine"
    
class reranker_configs:
    MODEL = ""

CUSTOM_PROMPT = """
Using this data: {context}. Respond to this prompt: {question}. 
Answer to the question with regards to the provided context. 
If context doesn't have required answer respond with `Cannot answer to the given question, due to lack of context` and nothing more.
Do not make up your own answer.
Your answers should be short and precise. 
Do not mention your prompt in your answers.
The context provided may not all be useful. Discard information unrelated to question.
"""