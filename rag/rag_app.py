from flask import Flask, render_template, request, jsonify
import os
from rag_utils import (
    load_data,
    get_chunks,
    get_embeddings,
    retrieve_closest_chunk,
    get_rag_with_chunk,
)

app = Flask(__name__)

# Global variables to store data and embeddings
data_txt = None
chunks = None
embeddings = None


def initialize_rag():
    """Initialize RAG system with patient data"""
    global data_txt, chunks, embeddings

    # Load patient data
    data_file = "rag/data/patients.txt"
    if os.path.exists(data_file):
        data_txt = load_data(data_file)
        chunks = get_chunks(data_txt)
        embeddings = get_embeddings(chunks)
        print(f"RAG system initialized with {len(chunks)} chunks")
    else:
        print(f"Warning: {data_file} not found")


@app.route("/")
def index():
    """Main page with two-pane interface"""
    global data_txt
    if data_txt is None:
        initialize_rag()

    return render_template("index.html", data_content=data_txt or "No data loaded")


@app.route("/ask", methods=["POST"])
def ask_question():
    """Handle user questions and return RAG response"""
    global chunks, embeddings

    try:
        data = request.get_json()
        question = data.get("question", "").strip()

        if not question:
            return jsonify({"error": "Please enter a question"})

        if chunks is None or embeddings is None:
            return jsonify({"error": "RAG system not initialized"})

        # Get closest chunk and generate response
        closest_chunk, similarity, chunk_index, _, _ = retrieve_closest_chunk(
            question, chunks, embeddings
        )

        if closest_chunk:
            rag_response, _, _ = get_rag_with_chunk(
                question, closest_chunk, chunk_index
            )

            return jsonify(
                {
                    "response": rag_response.strip(),
                    "similarity": float(similarity),
                    "chunk_index": int(chunk_index),
                }
            )
        else:
            return jsonify({"error": "No relevant information found"})

    except Exception as e:
        return jsonify({"error": f"Error processing question: {str(e)}"})


if __name__ == "__main__":
    initialize_rag()
    app.run(debug=True, port=5100)
