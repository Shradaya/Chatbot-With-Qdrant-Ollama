class app:
    title = "Nepal's Constitution"

class ollama_configs:
    # # Gemma2, 2B
    model_name = "gemma2:2b"
    vector_size = "2304"
    answer_key = "text"
    re_ranker = ""
    
    # # # PHI 3, 3.8B
    # model_name = "phi3:3.8B" # 3.8 B
    # vector_size = "3072"
    # answer_key = "text"
    
    # # # llama 3.1, 8B
    # model_name = "llama3.1"
    # vector_size = "4096"
    # answer_key = "text"
    
class file_paths:
    new = "./pdfs/"
    archive = "./archived_pdfs/"
    output_file_path = "./answers_collection/"
    output_file_name = f"{ollama_configs.model_name.replace('.', '-').replace(':', '-')}.csv"

class qdrant_configs:
    HOST = "localhost"
    PORT = "6333"
    URI = f"http://{HOST}:{PORT}/"
    COLLECTION = f"constitution-{ollama_configs.model_name.replace('.', '-').replace(':', '-')}"
    K = 2
    VECTOR_SIZE = ollama_configs.vector_size
    DISTANCE = "Cosine"
    
class reranker_configs:
    MODEL = ""
    
# CUSTOM_PROMPT = """
# The provided document is the Constitution of Nepal 2072. This document encompasses all the legal frameworks, guidelines, and principles governing the country. It defines the structure of the state, the distribution of powerts between different level of government, and the funcamental rights and duties of citizens.

# When answering quqestions based on this document, please follow these guidelines:
# - Be precise and concise in your responses.
# - Ensure that the information is accurate and directly relevant to the question.
# - highlight key articles, sections, or provisions where applicable.
# - Provide context to your answers when nexessary, explaining the implications or importance of specific provisions
# - Maintain a neutral and informative tone, avoiding any personal opinions or interpretations.
# The goal is to provide clear and informative answers that help the user understand the specific aspects of the constitution of nepal 2072.

# Remember to always base your answers on the {context} provided and address the specific {question} asked. Keep your answers short.
# """
CUSTOM_PROMPT = "Using this data: {context}. Respond to this prompt: {question}"