from pymilvus import MilvusClient

client = MilvusClient("milvus_demo.db")
print(client.list_collections())
