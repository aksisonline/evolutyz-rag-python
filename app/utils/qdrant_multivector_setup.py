
import os
from qdrant_client import QdrantClient, models

class QdrantMultivectorSetup:
    def __init__(self, collection_name=None, qdrant_url=None, qdrant_api=None):
        self.collection_name = collection_name or os.getenv("COLLECTION_NAME", "dense_multivector_demo")
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.qdrant_api = qdrant_api or os.getenv("QDRANT_API", None)
        self.client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api
        )

    def create_collection(self, dense_size=384, colbert_size=128):
        vectors_config = {
            "dense": models.VectorParams(
                size=dense_size,
                distance=models.Distance.COSINE
            ),
            "colbert": models.VectorParams(
                size=colbert_size,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
                hnsw_config=models.HnswConfigDiff(m=0)
            )
        }
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_config
        )
        print(f"Collection '{self.collection_name}' created with dense and colbert multivector configs.")

if __name__ == "__main__":
    setup = QdrantMultivectorSetup()
    setup.create_collection()
