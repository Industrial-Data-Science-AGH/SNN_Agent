import chromadb

client = chromadb.PersistentClient(path="chroma")
coll = client.get_or_create_collection("docs")
coll.add(ids=["1"], documents=["hello world"])
print("Ingested 1 doc")
