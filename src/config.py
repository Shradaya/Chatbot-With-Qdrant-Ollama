class app:
    title = "Test Application"

class ollama_configs:
    # model_name = "llama3"
    # model_name = "qwen2:0.5b"
    # model_name = "qwen:0.5b"
    # model_name = "starcoder:1b"
    model_name = "llama3.1"
    
class file_paths:
    new = "./pdfs/"
    archive = "./archived_pdfs/"

class qdrant_configs:
    HOST = "localhost"
    PORT = "6333"
    URI = f"http://{HOST}:{PORT}/"
    COLLECTION = "tft"
    K = 5
    VECTOR_SIZE = "4096"
    DISTANCE = "Cosine"
    
CUSTOM_PROMPT = """
The provided document is the Constitution of Nepal 2072. This document encompasses all the legal frameworks, guidelines, and principles governing the country. It defines the structure of the state, the distribution of powerts between different level of government, and the funcamental rights and duties of citizens.

When answering quqestions based on this document, please follow these guidelines:
- Be precise and concise in your responses.
- Ensure that the information is accurate and directly relevant to the question.
- highlight key articles, sections, or provisions where applicable.
- Provide context to your answers when nexessary, explaining the implications or importance of specific provisions
- Maintain a neutral and informative tone, avoiding any personal opinions or interpretations.
The goal is to provide clear and informative answers that help the user understand the specific aspects of the constitution of nepal 2072.

Remember to always base your answers on the {context} provided and address the specific {question} asked.
"""
