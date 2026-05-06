from pymilvus import MilvusClient

client = MilvusClient("milvus_demo.db")  # or "/home/ec2-user/experiment/milvus_demo.db"
col = "fsdd_audio_vector_1024"

# sanity check
print(client.describe_collection(col))

rows = client.query(
    collection_name=col,
    filter="id >= 0",
    output_fields=["vector"],
    limit=5
)

for r in rows:
    print(r["vector"])          # full embedding (1024 floats)
