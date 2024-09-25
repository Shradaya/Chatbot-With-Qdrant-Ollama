import numpy as np
from tqdm import tqdm
from ..config import qdrant_configs
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
# from langchain_qdrant import Qdrant as LangChainQdrant
from sklearn.metrics.pairwise import cosine_similarity
from qdrant_client.http import models
from ..utils import remove_stop_words

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

    # def initialize_vector_store(self):
    #     self.vector_store = LangChainQdrant(
    #         client = self.client,
    #         collection_name = qdrant_configs.COLLECTION,
    #         embeddings = self.embedder
    #     )

    
    def insert_data_to_qdrant(self, data_items: list[dict]):
        # COUNT NUMBER OF DATA POINTS
        # valu = 0
        # for val in data_items:
        #     valu += len(val['chunks'])
        # print(valu)
        data_items = tqdm(data_items, desc = "Loading Embedding")
        for data_item in data_items:
            points = []
            title = data_item.get('title')
            sub_titles = data_item.get('sub_titles', [])
            data_list = data_item['chunks']
            
            embeddings = self.embedder._embed(data_list)
            
            for i, (sub_title, data, embedding) in enumerate(zip(sub_titles, data_list, embeddings)):
                if not all(isinstance(value, float) for value in embedding):
                    raise ValueError(f"Invalid embedding: {embedding}")
                points.append(PointStruct(id=i, 
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
    
    def rerank_documents(self, question, retrieved_docs, top_n=5):
        if not self.reranker:
            return retrieved_docs

        question_embedding = self.reranker._embed([question])[0]
        doc_embeddings = [doc.vector for doc in retrieved_docs]
        similarities = cosine_similarity([question_embedding], doc_embeddings)[0]
        
        ranked_indices = np.argsort(similarities)[::-1]
        ranked_docs = [retrieved_docs[i] for i in ranked_indices[:top_n]]
        return ranked_docs
    
    def search_in_qdrant(self, query, top_k = qdrant_configs.K):
        query_embedding = self.embedder._embed([query])[0]
        filter_words = remove_stop_words(query)
        
        conditions = [models.FieldCondition(key='metadata.sub_title', 
                                            match = models.MatchValue(value=word)) 
                      for word in filter_words]
        or_filter = models.Filter(should=conditions)

        search_result = self.client.search(
            collection_name = qdrant_configs.COLLECTION,
            query_vector = query_embedding,
            limit = top_k,
            query_filter = or_filter,
            with_payload = True,
            with_vectors = True,
            # score_threshold = 0.7
        )

        return search_result
