import chromadb
client = chromadb.PersistentClient(path="chroma")
coll = client.get_or_create_collection("docs")
print(coll.query(query_texts=["hello"], n_results=1))
