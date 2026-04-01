# store_index.py
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema import Document  # if you are wrapping text+metadata
import os

# ----------------------------
# 1️⃣ Load your documents
# ----------------------------
# Example: list of Document objects or strings
# Replace this with your actual document loading
documents = [
    Document(page_content="Document 1 text", metadata={"source": "file1.txt"}),
    Document(page_content="Document 2 text", metadata={"source": "file2.txt"}),
    # ... add all your documents here
]

print(f"Total documents: {len(documents)}")

# ----------------------------
# 2️⃣ Setup embeddings
# ----------------------------
# Using a lightweight, fast model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda"} if os.environ.get("CUDA_VISIBLE_DEVICES") else {}
)

# ----------------------------
# 3️⃣ Create or load Chroma collection
# ----------------------------
collection_name = "my_collection"
vector_store = Chroma(collection_name=collection_name, embedding_function=embeddings)

# ----------------------------
# 4️⃣ Add documents in safe batches
# ----------------------------
MAX_BATCH = 1000  # safe batch size for Chroma (well below 5461 limit)
for i in range(0, len(documents), MAX_BATCH):
    batch = documents[i:i+MAX_BATCH]
    vector_store.add_documents(batch)
    print(f"Inserted batch {i} to {i+len(batch)-1}")

print("✅ All documents indexed successfully!")

#  ----------------------------
# 5️⃣ Query the vector store
# ----------------------------
query = "Some text you want to search"
results = vector_store.similarity_search(query, k=3)  # top 3 matches

print("\n🔍 Search Results:")
for r in results:
    print(r.page_content, r.metadata)