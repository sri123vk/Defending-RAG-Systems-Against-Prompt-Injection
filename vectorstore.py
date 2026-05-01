import chromadb
from sentence_transformers import SentenceTransformer
from corpus import clean_docs, poisoned_docs, poisoned_labels

def build_vectorstore():
    print("Loading embedder...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print("Setting up ChromaDB...")
    client = chromadb.Client()
    try:
        client.delete_collection("rag_corpus")
    except:
        pass
    collection = client.create_collection("rag_corpus")

    print("Embedding clean documents...")
    clean_embeddings = embedder.encode(clean_docs).tolist()
    collection.add(
        documents=clean_docs,
        embeddings=clean_embeddings,
        ids=[f"clean_{i}" for i in range(len(clean_docs))],
        metadatas=[{"type": "clean"} for _ in clean_docs]
    )

    print("Embedding poisoned documents...")
    poisoned_embeddings = embedder.encode(poisoned_docs).tolist()
    collection.add(
        documents=poisoned_docs,
        embeddings=poisoned_embeddings,
        ids=[f"poisoned_{i}" for i in range(len(poisoned_docs))],
        metadatas=[{"type": "poisoned", "attack_type": label}
                   for label in poisoned_labels]
    )

    print(f"\nVectorstore built successfully!")
    print(f"Total documents stored: {collection.count()}")
    return collection, embedder


def retrieve(query, collection, embedder, top_k=3):
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]
    return docs, metas, distances


if __name__ == "__main__":
    collection, embedder = build_vectorstore()

    print("\n--- Test Retrieval ---")
    test_query = "How do distributed systems handle load balancing?"
    docs, metas, distances = retrieve(test_query, collection, embedder, top_k=3)
    print(f"Query: '{test_query}'")
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
        print(f"[{i+1}] Type: {meta['type']} | Distance: {dist:.4f}")
        print(f"     {doc[:100]}...")
        print()
