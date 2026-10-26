"""
app.py
------
Flask backend for the RAG chatbot. Serves the frontend and exposes API
endpoints for document upload and querying. Same route layout as the
reference app: /, /api/status, /api/upload, /api/query, /api/reset.

Run with:
    put GROQ_API_KEY=... in a .env file next to this file
    python app.py
Then open http://localhost:5000
"""

import json
import os
import sys

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

from rag.embeddings import EmbeddingManager
from rag.vector_store import VectorStore
from rag.retriever import RAGRetriever
from rag.pipeline import get_llm, rag_advanced
from rag.history import ChatHistoryManager
from loaders.document_loader import load_document, SUPPORTED_EXTENSIONS, CHUNK_SIZE, CHUNK_OVERLAP

# ─────────────────────────────────────────────
# Flask app configuration
# ─────────────────────────────────────────────

app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
VECTOR_STORE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "vector_store")
HISTORY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "chat_history.json")
INDEX_SETTINGS_PATH = os.path.join(VECTOR_STORE_DIR, "index_settings.json")

# Allowed range for the chunk size a visitor can request (in words)
MIN_CHUNK_SIZE = 50
MAX_CHUNK_SIZE = 1000

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB max

# ─────────────────────────────────────────────
# Initialize RAG components once at startup
# ─────────────────────────────────────────────

print("=" * 60)
print("Initializing RAG Chatbot...")
print("=" * 60)

embedding_manager = EmbeddingManager()
vector_store = VectorStore(persist_directory=VECTOR_STORE_DIR, dim=embedding_manager.dim)
retriever = RAGRetriever(vector_store, embedding_manager)
history_manager = ChatHistoryManager(persist_path=HISTORY_PATH)

# How many recent turns to feed back to the model as conversational context
HISTORY_WINDOW = 5

_llm = None


def get_cached_llm():
    global _llm
    if _llm is None:
        _llm = get_llm()
    return _llm


print("RAG Chatbot initialized successfully!\n")


# ─────────────────────────────────────────────
# Index settings: which chunk size / overlap built the current index
# ─────────────────────────────────────────────

def read_index_settings():
    """Settings the current index was built with, or None if unknown."""
    try:
        with open(INDEX_SETTINGS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write_index_settings(settings: dict) -> None:
    with open(INDEX_SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)


def clear_index_settings() -> None:
    if os.path.exists(INDEX_SETTINGS_PATH):
        os.remove(INDEX_SETTINGS_PATH)


def parse_chunk_settings(form):
    """Read chunk_size / chunk_overlap from the upload form. Returns (size, overlap).
    Missing values fall back to the loader defaults. Raises ValueError for bad input."""
    try:
        size = int(form.get("chunk_size") or CHUNK_SIZE)
        overlap = int(form.get("chunk_overlap") or CHUNK_OVERLAP)
    except (TypeError, ValueError):
        raise ValueError("Chunk size and overlap must be whole numbers.")
    if not MIN_CHUNK_SIZE <= size <= MAX_CHUNK_SIZE:
        raise ValueError(f"Chunk size must be between {MIN_CHUNK_SIZE} and {MAX_CHUNK_SIZE} words.")
    overlap = max(0, min(overlap, size // 2))  # same rule the loader applies
    return size, overlap

# ─────────────────────────────────────────────
# Routes — Frontend
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")

# ─────────────────────────────────────────────
# API — Status
# ─────────────────────────────────────────────

@app.route("/api/status", methods=["GET"])
def status():
    count = vector_store.get_count()
    return jsonify({
        "status": "ready",
        "documents_loaded": count,
        "has_documents": count > 0,
        "index": read_index_settings() if count > 0 else None,
        "defaults": {"chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP},
    })

# ─────────────────────────────────────────────
# API — Upload document
# ─────────────────────────────────────────────

@app.route("/api/upload", methods=["POST"])
def upload_document():
    """
    Accepts a document, loads + chunks it, generates embeddings, stores in FAISS.
    Supported: PDF, TXT, MD, DOCX

    Optional form fields:
        chunk_size, chunk_overlap  words per chunk / words repeated between chunks
        keep_history = "1"         keep the chat (used when re-indexing the same file)
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    ext = os.path.splitext(filename)[1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    try:
        chunk_size, chunk_overlap = parse_chunk_settings(request.form)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    keep_history = request.form.get("keep_history") == "1"

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    try:
        documents = load_document(file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not documents:
            return jsonify({"error": "Could not extract text from the document"}), 400

        # Keep only the current upload — clear previous chunks. Chat history is cleared too,
        # unless this is a re-index of the same file (keep_history), where the old turns
        # are still about the right document and are useful for comparing settings.
        vector_store.reset()
        if not keep_history:
            history_manager.clear()

        texts = [doc.page_content for doc in documents]
        embeddings = embedding_manager.generate_embeddings(texts)
        vector_store.add_documents(documents, embeddings)

        settings = {
            "filename": filename,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "chunks": len(documents),
        }
        write_index_settings(settings)

        return jsonify({
            "success": True,
            "filename": filename,
            "chunks_loaded": len(documents),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "total_documents": vector_store.get_count(),
            "message": (f"Loaded '{filename}': {len(documents)} chunk(s) of up to {chunk_size} words, "
                        f"{chunk_overlap} overlap."),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

# ─────────────────────────────────────────────
# API — Chat history
# ─────────────────────────────────────────────

@app.route("/api/history", methods=["GET"])
def get_history():
    """Returns the full persisted conversation, so the frontend can restore
    it on page load."""
    return jsonify({"history": history_manager.get_all()})

# ─────────────────────────────────────────────
# API — Query
# ─────────────────────────────────────────────

@app.route("/api/query", methods=["POST"])
def query():
    """Body: { "query": "...", "top_k": 5 (optional) }"""
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "No query provided"}), 400

    user_query = data["query"].strip()
    if not user_query:
        return jsonify({"error": "Query cannot be empty"}), 400

    if vector_store.get_count() == 0:
        return jsonify({
            "error": "No documents loaded. Please upload a document first.",
            "answer": None,
        }), 400

    if not os.getenv("GROQ_API_KEY"):
        return jsonify({"error": "GROQ_API_KEY not found. Set it in .env."}), 400

    top_k = int(data.get("top_k", 5))

    try:
        llm = get_cached_llm()
        recent_history = history_manager.get_recent(HISTORY_WINDOW)

        result = rag_advanced(
            query=user_query,
            retriever=retriever,
            llm=llm,
            top_k=top_k,
            min_score=0.2,
            return_context=False,
            chat_history=recent_history,
        )

        # Remember which chunk settings this answer was produced under
        current = read_index_settings() or {}
        index_tag = {
            "chunk_size": current.get("chunk_size"),
            "chunk_overlap": current.get("chunk_overlap"),
            "chunks": current.get("chunks"),
        } if current else None

        # Persist this turn so it survives restarts and informs future follow-ups
        history_manager.add(user_query, result["answer"], result["sources"], index=index_tag)

        return jsonify({
            "answer": result["answer"],
            "sources": result["sources"],
            "confidence": result["confidence"],
            "query": user_query,
            "index": index_tag,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─────────────────────────────────────────────
# API — Reset vector store
# ─────────────────────────────────────────────

@app.route("/api/reset", methods=["DELETE"])
def reset():
    try:
        vector_store.reset()
        history_manager.clear()
        clear_index_settings()
        return jsonify({"success": True, "message": "Vector store and chat history cleared. Ready for new documents."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\nStarting RAG Chatbot server...")
    print("Open http://localhost:5000 in your browser\n")
    # Local use only: listen on this computer (not the whole network) and keep debug off
    # unless you ask for it with FLASK_DEBUG=1.
    app.run(debug=os.getenv("FLASK_DEBUG") == "1", host="127.0.0.1", port=5000)
