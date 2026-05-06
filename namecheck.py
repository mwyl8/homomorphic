from pymilvus import MilvusClient

client = MilvusClient("milvus_demo.db")
collection_name = "yamnet_embeddings"

schema = client.describe_collection(collection_name)
print("Fields:", schema["fields"])
