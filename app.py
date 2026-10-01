import os

import streamlit as st
import pymupdf
import faiss
import numpy as np

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai


# ==========================================
# 1. Load Gemini API key
# ==========================================

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    st.error(
        "GEMINI_API_KEY not found. "
        "Please add it to your .env file."
    )
    st.stop()


# ==========================================
# 2. Create Gemini client
# ==========================================

client = genai.Client(
    api_key=gemini_api_key
)


# ==========================================
# 3. Load BGE-M3
# ==========================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "BAAI/bge-m3"
    )


model = load_embedding_model()


# ==========================================
# 4. Streamlit UI
# ==========================================

st.title("Resume RAG")

st.write(
    "Upload a resume and ask questions about it."
)


# ==========================================
# 5. Upload Resume
# ==========================================

file = st.file_uploader(
    "Upload Resume",
    type=["pdf"]
)


if file:

    # ==========================================
    # 6. Extract text from PDF
    # ==========================================

    pdf = pymupdf.open(
        stream=file.read(),
        filetype="pdf"
    )

    text = ""

    for page in pdf:
        text += page.get_text()

    pdf.close()


    if not text.strip():

        st.error(
            "No text could be extracted from the PDF."
        )

        st.stop()


    st.success(
        "Resume loaded successfully!"
    )


    # ==========================================
    # 7. Split text into chunks
    # ==========================================

    words = text.split()

    chunks = []

    chunk_size = 30
    overlap = 10

    step = chunk_size - overlap

    for i in range(
        0,
        len(words),
        step
    ):

        chunk = " ".join(
            words[i:i + chunk_size]
        )

        if chunk.strip():

            chunks.append(chunk)


    st.write(
        "Number of chunks:",
        len(chunks)
    )


    # ==========================================
    # 8. Create BGE-M3 embeddings
    # ==========================================

    with st.spinner(
        "Creating BGE-M3 embeddings..."
    ):

        embeddings = model.encode(
            chunks,
            normalize_embeddings=True,
            show_progress_bar=False
        )


    embeddings = np.array(
        embeddings,
        dtype="float32"
    )


    # ==========================================
    # 9. Create FAISS index
    # ==========================================

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(
        embeddings
    )


    st.success(
        "Resume indexed successfully!"
    )


    # ==========================================
    # 10. Ask question
    # ==========================================

    question = st.text_input(
        "Ask something about your resume"
    )


    if st.button("Ask") and question:


        # ==========================================
        # 11. Convert question to BGE-M3 embedding
        # ==========================================

        with st.spinner(
            "Searching resume..."
        ):

            question_embedding = model.encode(
                [question],
                normalize_embeddings=True,
                show_progress_bar=False
            )


        question_embedding = np.array(
            question_embedding,
            dtype="float32"
        )


        # ==========================================
        # 12. Search FAISS
        # ==========================================

        scores, ids = index.search(
            question_embedding,
            3
        )


        # ==========================================
        # 13. Get top 3 chunks
        # ==========================================

        retrieved_chunks = []

        for i in ids[0]:

            if i != -1:

                retrieved_chunks.append(
                    chunks[i]
                )


        context = "\n\n".join(
            retrieved_chunks
        )


        # ==========================================
        # 14. Create RAG prompt
        # ==========================================

        prompt = f"""
You are a resume question-answering assistant.

Answer the user's question using ONLY the
information provided in the resume context.

Do not use outside knowledge.

If the answer is not present in the
resume context, answer exactly:

"Not mentioned in the resume."

Resume Context:
{context}

Question:
{question}
"""


        # ==========================================
        # 15. Generate answer using Gemini
        # ==========================================

        with st.spinner(
            "Generating answer..."
        ):

            try:

                response = client.interactions.create(
                    model="gemini-3.1-flash-lite",
                    input=prompt
                )

                answer = response.output_text


            except Exception as e:

                answer = (
                    f"Gemini error: {str(e)}"
                )


        # ==========================================
        # 16. Display answer
        # ==========================================

        st.subheader("Answer")

        st.write(
            answer
        )


        # ==========================================
        # 17. Show retrieved chunks
        # ==========================================

        with st.expander(
            "View Retrieved Chunks"
        ):

            for number, chunk in enumerate(
                retrieved_chunks,
                start=1
            ):

                st.write(
                    f"### Chunk {number}"
                )

                st.write(chunk)

                st.write("---")