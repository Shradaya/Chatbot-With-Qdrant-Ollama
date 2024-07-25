from ..config import CUSTOM_PROMPT
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


def get_qa_chain(llm, 
                 retriever, 
                 return_source_documents = False, 
                 chain_type = "stuff"
                ):
    PROMPT = PromptTemplate(
        template = CUSTOM_PROMPT,
        input_variables = ["context", "question"]
    )
    return RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = chain_type,
        retriever = retriever,
        return_source_documents = return_source_documents,
        chain_type_kwargs = {
            "prompt": PROMPT
            }
    )