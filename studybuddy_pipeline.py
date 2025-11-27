import os
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

# --- Configuration ---
folder_path = "study_materials"
output_folder = "chunks"
os.makedirs(output_folder, exist_ok=True)

# Load FREE local embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Helper functions ---
def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

all_embeddings = []

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    if filename.lower().endswith(".pdf"):
        text = read_pdf(file_path)
    elif filename.lower().endswith(".txt"):
        text = read_txt(file_path)
    else:
        continue

    chunks = text_splitter.split_text(text)

    for i, chunk in enumerate(chunks, start=1):
        chunk_file = os.path.join(output_folder, f"{filename}_chunk_{i}.txt")

        with open(chunk_file, "w", encoding="utf-8") as f:
            f.write(chunk)

        # Local offline embedding
        embedding_vector = model.encode(chunk).tolist()

        all_embeddings.append({
            "file": filename,
            "chunk_file": chunk_file,
            "chunk_index": i,
            "text": chunk,
            "embedding": embedding_vector
        })

    print(f"{filename} → {len(chunks)} chunks created.")

# Save JSON
with open(os.path.join(output_folder, "embeddings.json"), "w", encoding="utf-8") as f:
    json.dump(all_embeddings, f)

print("✅ ALL DONE! Offline embeddings created.")

