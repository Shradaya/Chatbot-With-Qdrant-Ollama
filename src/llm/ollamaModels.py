import ollama
from ..config import ollama_configs
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

llm = Ollama(
    model = ollama_configs.model_name
)
embedder = OllamaEmbeddings(
    model = ollama_configs.model_name,
)

if ollama_configs.re_ranker:
    reranker = OllamaEmbeddings(
        model = ollama_configs.re_ranker
    )
else:
    reranker = None
    
def check(document, claim):
    prompt = f"Document: {document}\nClaim: {claim}. Respond with `Yes` or `No`"
    response = ollama.generate(
        model="bespoke-minicheck", prompt=prompt, options={"num_predict": 2, "temperature": 0.0}
    )
    return response["response"].strip()
