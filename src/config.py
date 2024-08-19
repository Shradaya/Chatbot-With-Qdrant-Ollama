class app:
    title = "Test Application"

class ollama_configs:
    model_name = "qwen:0.5b"
    # model_name = "qwen2:0.5b"
    # model_name = "starcoder:1b"
    # model_name = "llama3.1"
    
class file_paths:
    new = "./pdfs/"
    archive = "./archived_pdfs/"

class qdrant_configs:
    HOST = "localhost"
    PORT = "6333"
    URI = f"http://{HOST}:{PORT}/"
    COLLECTION = f"constitution-{ollama_configs.model_name.replace('.', '-').replace(':', '-')}"
    
CUSTOM_PROMPT = """
You are an AI assistant specialized in the Critical perspective in accounting.
Do not make up or infer information
Remember to always base your answers on the {context} provided and address the specific {question} asked.
Be concise in your responses while ensuring accuracy and completeness.
"""
