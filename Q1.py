import nltk
import string
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os 

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords and punctuation, also eliminate empty space tokens
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation and token.strip()]
    return tokens

def build_positional_index(data):
    positional_index = defaultdict(lambda: defaultdict(list))
    for doc_id, text in enumerate(data):
        tokens = preprocess_text(text)
        for pos, term in enumerate(tokens):
            positional_index[term][doc_id].append(pos)
    return positional_index

def phrase_query_search(phrase, positional_index):
    if len(phrase.split()) > 5:
        return "Query too long"
    
    phrase_tokens = preprocess_text(phrase)
    postings = [positional_index[token] for token in phrase_tokens]
    
    # Find common document IDs
    common_docs = set(postings[0]).intersection(*postings[1:])
    
    # Check positional criteria for phrase query
    results = []
    for doc_id in common_docs:
        # Get positions for each term in the document
        positions = [postings[i][doc_id] for i in range(len(phrase_tokens))]
        # Check if positions satisfy the phrase query
        if all(any(p2 - p1 == 1 for p1, p2 in zip(positions[i], positions[i + 1])) for i in range(len(positions) - 1)):
            results.append(doc_id)
    
    return results



# 
def read_data(folder_path):
    documents = []
    document_names = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='iso-8859-1') as file:
                documents.append(file.read())
                document_names.append(filename)
    return documents, document_names

def display_search_results(phrase, search_results, document_names):
    print(f"Search results IDs: {search_results}")  # Debug: Show which IDs are being accessed
    if search_results:
        print(f"The phrase '{phrase}' is found in the following document(s):")
        for doc_id in search_results:
            # Ensure the doc_id is within range
            if doc_id >= 0 and doc_id < len(document_names):
                print(f"- {document_names[doc_id]}")
            else:
                print(f"Error: Document ID {doc_id} is out of valid range.")
    else:
        print(f"The phrase '{phrase}' was not found in any document.")
# Paths
data_folder_path = '/Users/ayushchhetri/Desktop/cp423/CP423/Assignment2/data'  # Adjust this path

# Read and process the data
data, document_names = read_data(data_folder_path)
positional_index = build_positional_index(data)

# Search for a phrase and display results
phrase = "little pigs"  # Example phrase
search_results = phrase_query_search(phrase, positional_index)
display_search_results(phrase, search_results, document_names)