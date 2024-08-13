from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from ..config import qdrant_configs, ollama_configs
from langchain_community.embeddings import OllamaEmbeddings
from langchain_qdrant import Qdrant as LangChainQdrant

class qdrant_connection:
    def __init__(self, embedder):
        self.client = QdrantClient(
            host=qdrant_configs.HOST,
            port=qdrant_configs.PORT
        )
        self.embedder = embedder
        self.vector_store = self.initialize_vector_store()
        
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

    def initialize_vector_store(self):
        self.vector_store = LangChainQdrant(
            client = self.client,
            collection_name = qdrant_configs.COLLECTION,
            embeddings = self.embedder
        )

    
    def insert_data_to_qdrant(self, data_items: list[str]):
        points = []
        embeddings = self.embedder._embed(data_items)
        for i, (data, embedding) in enumerate(zip(data_items, embeddings)):
            if not all(isinstance(value, float) for value in embedding):
                raise ValueError(f"Invalid embedding: {embedding}")
            points.append(PointStruct(id=i, vector=embedding, payload={'text': data}))
        
        self.client.upsert(collection_name=qdrant_configs.COLLECTION, points=points)
        
    def search_in_qdrant(self, query, top_k = qdrant_configs.K):
        query_embedding = self.embedder._embed([query])[0]

        search_result = self.client.search(
            collection_name = qdrant_configs.COLLECTION,
            query_vector = query_embedding,
            limit = top_k
        )
        return search_result
