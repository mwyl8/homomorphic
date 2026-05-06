from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530") 
schema = client.describe_collection("audio_vectors")
print(schema)

