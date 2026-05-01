import os
import anthropic
import chromadb
from sentence_transformers import SentenceTransformer

# Check API key
api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    print("API key NOT set!")
    exit(1)
else:
    print(f"API key found: {api_key[:10]}...")

# Test Anthropic
client = anthropic.Anthropic(api_key=api_key)
message = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=50,
    messages=[{"role": "user", "content": "Say hello in one sentence."}]
)
print("Claude says:", message.content[0].text)
print("Anthropic OK")

# Test ChromaDB
db = chromadb.Client()
collection = db.create_collection("test")
print("ChromaDB OK")

# Test embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")
vec = embedder.encode("hello world")
print(f"Embedder OK - dim: {len(vec)}")
