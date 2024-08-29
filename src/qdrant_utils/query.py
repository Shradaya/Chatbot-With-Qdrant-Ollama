from langchain_qdrant import Qdrant
from ..config import qdrant_configs

def insert_into_db(embedder, file_contents_dict_lst = []):
    texts = []
    metadatas = []
    for item in file_contents_dict_lst:
        texts.append(item.get('chunk'))
        metadatas.append({
                "title": item.get('title')
            })
    
    _ = Qdrant.from_texts(
        texts, 
        embedder,
        url = qdrant_configs.URI, 
        collection_name = qdrant_configs.COLLECTION,
        force_recreate = True,
        metadatas = metadatas
    )
    
def get_retriever(embedder, search_type = "mmr", search_item_count = 5):
    qdrant = Qdrant.from_existing_collection(
        embedding = embedder,
        collection_name = qdrant_configs.COLLECTION,
        url = qdrant_configs.URI,
        metadata_payload_key='title'
    )
    return qdrant.as_retriever(search_type = search_type, search_kwargs={"k": search_item_count})