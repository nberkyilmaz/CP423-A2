import os
import numpy as np
import math
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import Q1

# Helper function to compute the document frequency for each term
def compute_document_frequencies(positional_index):
    # Initialize a dictionary to store document frequencies
    document_frequencies = {}
    # Loop through each term in the positional index
    for term, postings_list in positional_index.items():
        # The document frequency of a term is the number of documents it appears in
        document_frequencies[term] = len(postings_list)
    return document_frequencies

# Function to calculate IDF using the given formula
def compute_idf(document_frequencies, total_documents):
    # Initialize a dictionary to store IDFs
    idfs = {}
    # Loop through each term in the document frequencies
    for term, df in document_frequencies.items():
        # Calculate IDF using the given formula
        idfs[term] = math.log(total_documents / (df + 1))
    return idfs

# Helper function to calculate TF using different weighting schemes
def compute_tf(term, doc_id, positional_index, scheme):
    # Get the raw count of the term in the specified document
    raw_count = len(positional_index[term][doc_id])
    # Calculate TF based on the specified scheme
    if scheme == "binary":
        return 1 if raw_count > 0 else 0
    elif scheme == "raw_count":
        return raw_count
    elif scheme == "term_frequency":
        total_terms = sum(len(positional_index[term][doc_id]) for term in positional_index.keys())
        return raw_count / total_terms if total_terms > 0 else 0
    elif scheme == "log_normalization":
        return math.log(1 + raw_count)
    elif scheme == "double_normalization":
        max_tf = max(len(positional_index[term][doc_id]) for term in positional_index.keys())
        return 0.5 + 0.5 * (raw_count / max_tf) if max_tf > 0 else 0

# Main function to generate TF-IDF matrix and handle query processing
def generate_and_query_tfidf(data, document_names, query, scheme="binary"):
    # Calculate the total number of documents
    total_documents = len(data)
    # Build a positional index from the data
    positional_index = build_positional_index(data)
    # Compute document frequencies for each term
    document_frequencies = compute_document_frequencies(positional_index)
    # Calculate IDFs for each term
    idfs = compute_idf(document_frequencies, total_documents)
    
    # Initialize a matrix to store TF-IDF values for each document and term
    tfidf_matrix = np.zeros((total_documents, len(positional_index)))
    # Populate the TF-IDF matrix
    for col, term in enumerate(positional_index.keys()):
        for doc_id in positional_index[term]:
            tf = compute_tf(term, doc_id, positional_index, scheme)
            tfidf_matrix[doc_id, col] = tf * idfs[term]

    # Preprocess query and create query vector
    query_terms = preprocess_text(query)
    query_vector = np.zeros(len(positional_index))
    for term in query_terms:
        if term in positional_index:
            index = list(positional_index.keys()).index(term)
            query_vector[index] = compute_tf(term, 0, {term: positional_index[term]}, scheme) * idfs[term] if term in idfs else 0

    # Compute cosine similarity between each document vector and the query vector
    cosine_similarities = np.dot(tfidf_matrix, query_vector) / (np.linalg.norm(tfidf_matrix, axis=1) * np.linalg.norm(query_vector) + 1e-10)
    
    # Get indices of the top 5 relevant documents based on cosine similarity
    top_documents_indices = np.argsort(cosine_similarities)[-5:][::-1]
    # Retrieve the names of the top documents
    top_documents = [document_names[index] for index in top_documents_indices if cosine_similarities[index] > 0]

    return top_documents

# Example usage
data_folder_path = 'path_to_your_data_folder'
data, document_names = read_data(data_folder_path)
query = "example query"
# Generate TF-IDF matrix and retrieve top documents for a given query
top_documents = generate_and_query_tfidf(data, document_names, query, scheme="binary")
for doc in top_documents:
    print(doc)
