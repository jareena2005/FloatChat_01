# app.py
from flask import Flask, render_template, request
from dotenv import load_dotenv
import os
from src.helpers import download_embeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import system_prompt

app = Flask(__name__)

# ----------------------------
# 1️⃣ Load environment variables
# ----------------------------
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# ----------------------------
# 2️⃣ Load embeddings
# ----------------------------
# Make sure this returns the same HuggingFace embeddings you used in store_index.py
embeddings = download_embeddings()

# ----------------------------
# 3️⃣ Load local Chroma vector store
# ----------------------------
# persist_directory must match where your store_index.py saved the index
docsearch = Chroma(
    persist_directory="db",   # folder where Chroma saved vectors
    embedding_function=embeddings
)

# ----------------------------
# 4️⃣ Setup retriever
# ----------------------------
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # top 3 relevant documents
)

# ----------------------------
# 5️⃣ Setup Groq LLM
# ----------------------------
chatModel = ChatGroq(
    model="llama-3.1-8b-instant",  # or any Groq model you have access to
    temperature=0,
)

# ----------------------------
# 6️⃣ Setup Prompt and RAG chain
# ----------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# ----------------------------
# 7️⃣ Flask routes
# ----------------------------
@app.route("/")
def index():
    return render_template("chat.html")  # your HTML chat frontend

@app.route("/get-floats", methods=["GET"])
def get_floats():
    """Return float location data dynamically from ChromaDB"""
    import re
    
    # Query ChromaDB for float/location data
    results = docsearch.similarity_search("float latitude longitude coordinates location", k=10)
    
    floats = []
    for doc in results:
        content = doc.page_content
        
        # Extract latitude and longitude patterns from text
        # Look for patterns like "lat: 19.0760" or "latitude 19.0760"
        lat_pattern = r'(?:lat|latitude)[:\s]+(-?\d+\.?\d*)'
        lon_pattern = r'(?:lon|longitude)[:\s]+(-?\d+\.?\d*)'
        
        lat_match = re.search(lat_pattern, content, re.IGNORECASE)
        lon_match = re.search(lon_pattern, content, re.IGNORECASE)
        
        if lat_match and lon_match:
            try:
                lat = float(lat_match.group(1))
                lon = float(lon_match.group(1))
                
                # Avoid duplicates
                if not any(f["lat"] == lat and f["lon"] == lon for f in floats):
                    floats.append({
                        "lat": lat,
                        "lon": lon,
                        "name": f"Float {len(floats) + 1}"
                    })
            except ValueError:
                continue
    
    # Return extracted floats or fallback to defaults if none found
    return floats if floats else [
        {"lat": 19.0760, "lon": 72.8777, "name": "Float 1"},
        {"lat": 28.7041, "lon": 77.1025, "name": "Float 2"}
    ]

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg") or request.json.get("msg")
    print("User:", msg)

    response = rag_chain.invoke({"input": msg})
    print("Full Response:", response)

    return str(response["answer"])

# ----------------------------
# 8️⃣ Run Flask app
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)