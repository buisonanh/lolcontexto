# Import the necessary libraries:
import os  # For reading files and managing paths
import numpy as np  # For performing mathematical operations
from scipy.sparse import lil_matrix  # For handling sparse matrices
from sklearn.decomposition import TruncatedSVD  # For Singular Value Decomposition (SVD)
from sklearn.metrics.pairwise import cosine_similarity  # For calculating cosine similarity between vectors
import streamlit as st

# Define the path to the corpus folder and obtain the list of text files
corpus_folder = "./corpus"
file_names = [f for f in os.listdir(corpus_folder) if f.endswith(".txt")]

# Initialize an empty list to store the words from the corpus
corpus = []

# Read each text file in the corpus folder and append the words to the corpus list
for file_name in file_names:
    file_path = os.path.join(corpus_folder, file_name)
    with open(file_path, "r", encoding="utf-8") as corpusFile:
        for linea in corpusFile:
            word_line = linea.strip().split()
            corpus.extend(word_line)

# Function to create a co-occurrence matrix from the corpus with a given window size
def create_co_occurrence_matrix(corpus, window_size=4):
    vocab = set(corpus)  # Create a set of unique words in the corpus
    word2id = {word: i for i, word in enumerate(vocab)}  # Create a word-to-index dictionary for the words
    id2word = {i: word for i, word in enumerate(vocab)}  # Create an index-to-word dictionary for the words
    matrix = lil_matrix((len(vocab), len(vocab)))  # Initialize an empty sparse matrix of size len(vocab) x len(vocab)

    # Iterate through the corpus to fill the co-occurrence matrix
    for i in range(len(corpus)):
        for j in range(max(0, i - window_size), min(len(corpus), i + window_size)):
            if i != j:
                matrix[word2id[corpus[i]], word2id[corpus[j]]] += 1

    return matrix, word2id, id2word

# Function to perform SVD on the co-occurrence matrix and reduce the dimensionality
def perform_svd(matrix, n_components=300):
    n_components = min(n_components, matrix.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components)
    return svd.fit_transform(matrix)

# Function to create word embeddings from the corpus using the co-occurrence matrix and SVD
def create_word_embeddings(corpus):
    matrix, word2id, id2word = create_co_occurrence_matrix(corpus)  # Create the co-occurrence matrix
    word_embeddings = perform_svd(matrix)  # Perform SVD on the matrix
    return word_embeddings, word2id, id2word

# Create the word embeddings from the given corpus
embeddings, word2id, id2word = create_word_embeddings(corpus)


# Function to calculate the cosine similarity between two word vectors
def get_word_similarity(embeddings, word2id, word1, word2):
    word1_vector = embeddings[word2id[word1]]  # Get the vector representation of word1
    word2_vector = embeddings[word2id[word2]]  # Get the vector representation of word2

    # Compute the cosine similarity between the two vectors
    similarity = cosine_similarity(word1_vector.reshape(1, -1), word2_vector.reshape(1, -1))

    return similarity[0][0]


# opening the file in read mode 
my_file = open("champ_list.txt", "r") 

# reading the file 
data = my_file.read() 

# replacing end splitting the text  
# when newline ('\n') is seen. 
champions = data.split(", ") 

my_file.close() 


# Initialize the session state to store the answer and guesses if not already done
if 'answer' not in st.session_state:
    st.session_state['answer'] = np.random.choice(champions)

if 'guesses' not in st.session_state:
    st.session_state['guesses'] = []


# Generate a random champion name to guess from the champions list
answer = st.session_state['answer']
# Create a list of tuples with champion names and their similarity scores to 'Aatrox'
similarity_scores = [(c, get_word_similarity(embeddings, word2id, answer, c)) for c in champions]
# Sort the list based on similarity scores in descending order
similarity_scores.sort(key=lambda x: x[1], reverse=True)
# Create a ranking dictionary
ranking = {champion: rank + 1 for rank, (champion, _) in enumerate(similarity_scores)}


# Get choice
user_choice = st.text_input("Enter a champion name").lower()

# Submit button
if st.button("Submit") or user_choice:
    if user_choice in champions:
        score = 100 - int(100*(ranking[user_choice]/len(ranking)))
        if user_choice in champions and (user_choice, score) not in st.session_state['guesses']:
            st.session_state['guesses'].append((user_choice, score))
        else:
            st.write(f"The champion '{user_choice}' is not valid.")

# Sort the guesses by rank before displaying
sorted_guesses = sorted(st.session_state['guesses'], key=lambda x: x[1], reverse=True)

if user_choice == answer:
    st.balloons()
    st.title("You guessed the correct champion!!")

# Display all guesses with progress bars
for guess, rank in sorted_guesses:
    st.progress(rank, text=guess)

# Button to print the answer
if st.button("Print Answer"):
    st.write(f"The correct champion is '{answer}'.")

# Button to reset the game
if st.button("Reset Game"):
    st.session_state['guesses'] = []
    st.session_state['answer'] = np.random.choice(champions)
    st.write("Game has been reset. Good luck!")

