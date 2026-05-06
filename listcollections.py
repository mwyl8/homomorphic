from pymilvus import MilvusClient

client = MilvusClient("milvus_demo.db")
print("Collections:", client.list_collections())
