import numpy as np
from tqdm import tqdm
from ..config import qdrant_configs
from qdrant_client.http import models
from ..utils import remove_stop_words
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from sklearn.metrics.pairwise import cosine_similarity

class qdrant_connection:
    def __init__(self, embedder, reranker = None):
        self.client = QdrantClient(
            host=qdrant_configs.HOST,
            port=qdrant_configs.PORT
        )
        self.embedder = embedder
        self.reranker = reranker
        # self.vector_store = self.initialize_vector_store()
        
        self.create_collection()
        
    def create_collection(self):
        collection_name = qdrant_configs.COLLECTION
        existing_collections = self.client.get_collections()
        if collection_name not in [existing.name for existing in existing_collections.collections]:
            self.client.create_collection(
                collection_name = collection_name,
                vectors_config = {
                    "size": qdrant_configs.VECTOR_SIZE,
                    "distance": qdrant_configs.DISTANCE
                }
            )
            print(f"Collection '{collection_name}' created.")
        else:
            # Collection already exists
            print(f"Collection '{collection_name}' already exists.")
    
    def insert_data_to_qdrant(self, data_items: list[dict]):
        data_items = tqdm(data_items, desc = "Loading Embedding")
        index = 1
        for data_item in data_items:
            points = []
            title = data_item.get('title')
            sub_titles = data_item.get('sub_titles', [])
            data_list = data_item['chunks']
            
            embeddings = self.embedder._embed(data_list)
            
            for sub_title, data, embedding in zip(sub_titles, data_list, embeddings):
                if not all(isinstance(value, float) for value in embedding):
                    raise ValueError(f"Invalid embedding: {embedding}")
                index += 1
                points.append(PointStruct(id=index, 
                                          vector=embedding, 
                                          payload = {
                                              'text': data, 
                                              "metadata": {
                                                  "title": title,
                                                  "sub_title": sub_title
                                              },
                                              "vectors": embedding
                                            }))
        
            self.client.upsert(collection_name=qdrant_configs.COLLECTION, points=points)
    
    def rerank_documents(self, question, retrieved_docs: list, top_k: int = qdrant_configs.K):
        if not self.reranker:
            return retrieved_docs

        reranked_score = self.reranker.compute_score([[question, answer] for answer in retrieved_docs])
        
        zipped = list(zip(reranked_score, retrieved_docs))
        sorted_zipped = sorted(zipped, key=lambda x: x[0], reverse=True)
        _, reranked_docs = zip(*sorted_zipped)

        if len(reranked_docs) > top_k:
            reranked_docs = reranked_docs[:top_k]

        return list(reranked_docs)
    
    def search_in_qdrant(self, query, retrive_count = qdrant_configs.RETRIEVE_COUNT):
        query_embedding = self.embedder._embed([query])[0]
        query = query.lower().replace('.', ' ').replace('?', '').replace(',', '').replace(';', '')
        filter_words = remove_stop_words(query)
        
        conditions = [models.FieldCondition(key='metadata.sub_title', 
                                            match = models.MatchValue(value=word)) 
                      for word in filter_words]
        or_filter = models.Filter(should=conditions)

        search_result = self.client.search(
            collection_name = qdrant_configs.COLLECTION,
            query_vector = query_embedding,
            query_filter = or_filter,
            limit = retrive_count,
            with_payload = True,
            score_threshold = 0.6
        )
        if not search_result:
            search_result = self.client.search(
                collection_name = qdrant_configs.COLLECTION,
                query_vector = query_embedding,
                limit = retrive_count,
                with_payload = True,
                with_vectors = True,
                score_threshold = 0.6
            )

        return search_result
