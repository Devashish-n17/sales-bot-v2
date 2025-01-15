import os
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from typing import List
from langchain.vectorstores import Chroma

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        return self.model.encode(documents)

    def embed_query(self, query: str) -> List[float]:
        return self.model.encode([query])[0]


model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embedding = SentenceTransformerEmbeddings(model_name)

def create_or_get_vector_storage(df, persist_directory, collection_name):
    chunks = []
    for index, row in df.iterrows():
        chunk = " | ".join([f"{col}: {row[col]}" for col in df.columns])
        chunks.append(chunk)

    documents = [Document(page_content=chunk, metadata={}) for chunk in chunks]
    
    if os.path.exists(persist_directory):
        vector_store = Chroma(persist_directory=persist_directory, collection_name=collection_name, embedding_function=embedding)
    else:
        vector_store = Chroma.from_documents(documents=documents, embedding=embedding, persist_directory=persist_directory, collection_name=collection_name)
    return vector_store 

# ===========================================================================

# from sentence_transformers import SentenceTransformer
# from langchain.vectorstores import Chroma
# from langchain.docstore.document import Document
# import os

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def sentence_transformer_embedding(texts):
    """Encodes text using sentence-transformers and returns list of embeddings."""
    return embedding_model.encode(texts).tolist()

def create_or_get_vector_storage(df, persist_directory, collection_name):
    chunks = []
    for index, row in df.iterrows():
        chunk = " | ".join([f"{col}: {row[col]}" for col in df.columns])
        chunks.append(chunk)

    documents = [Document(page_content=chunk, metadata={}) for chunk in chunks]

    if os.path.exists(persist_directory):
        vector_store = Chroma(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding_function=sentence_transformer_embedding
        )
    else:
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=sentence_transformer_embedding,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
    return vector_store

# ===========================================================================

# from sentence_transformers import SentenceTransformer
# sentences = ["This is an example sentence", "Each sentence is converted"]

# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# embeddings = model.encode(sentences)
# print(embeddings)

# ===========================================================================

# from sentence_transformers import SentenceTransformer, util
# import csv
# import torch
# import os

# model_name = 'paraphrase-multilingual-mpnet-base-v2'
# encoded_model_path = 'semantic_search_model.pt'
# dataset_path = 'dataset.csv'

# bi_encoder = SentenceTransformer(model_name)

# def encode_dataset():
#     passages = []
#     with open(dataset_path) as csv_file:
#         csv_reader = csv.reader(csv_file)
#         for row in csv_reader:
#             passages.append(row[0])

#     corpus_embeddings = bi_encoder.encode(
#         passages, batch_size=32, convert_to_tensor=True, show_progress_bar=True
#     )

#     torch.save(corpus_embeddings, encoded_model_path)
#     print("Dataset encoded and embeddings saved.")

# def perform_search(query):
#     # Load the pre-encoded embeddings
#     if not os.path.exists(encoded_model_path):
#         print("Error: The embeddings file does not exist. Please run the encoding part first.")
#         return

#     semantic_search_model = torch.load(encoded_model_path)

#     # Read the dataset for search
#     passages = []
#     with open(dataset_path) as csv_file:
#         csv_reader = csv.reader(csv_file)
#         for row in csv_reader:
#             passages.append(row[0])

#     # Encode the query
#     question_embedding = bi_encoder.encode(query, convert_to_tensor=True)

#     # Perform semantic search (top 3 results)
#     hits = util.semantic_search(question_embedding, semantic_search_model, top_k=3)
#     hits = hits[0]  # Get top results

#     # Format and display the results with their scores
#     result = {"search_results": [
#         {"score": hit['score'], "text": passages[hit['corpus_id']]} for hit in hits
#     ]}
#     return result

# # Main logic: Ask user whether to encode the dataset or perform a search
# def main():
#     action = input("Choose an action: (1) Encode dataset, (2) Perform search: ")

#     if action == '1':
#         # Encode dataset and save embeddings
#         encode_dataset()
#     elif action == '2':
#         query = input("Enter your search query: ")
#         results = perform_search(query)
#         if results:
#             print("Search Results:")
#             for result in results['search_results']:
#                 print(f"Score: {result['score']:.2f} - Text: {result['text']}")
#     else:
#         print("Invalid choice. Please choose 1 or 2.")

# if __name__ == "__main__":
#     main()
