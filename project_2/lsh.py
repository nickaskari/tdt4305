# This is the code for the LSH project of TDT4305

import configparser  # for reading the parameters file
import sys  # for system errors and printouts
from pathlib import Path  # for paths of files
import os  # for reading the input data
import time  # for timing
import numpy as np # for creating matrices or arrays
import random # for randomly generating a and b for hash functions
from itertools import combinations # for creating candidate pairs in lsh


'''
NOTEs:
- Can change data from BBC to test for shorter runtimes and easier visualization
- 
'''


# Global parameters
parameter_file = 'default_parameters.ini'  # the main parameters file
data_main_directory = Path('data')  # the main path were all the data directories are
parameters_dictionary = dict()  # dictionary that holds the input parameters, key = parameter name, value = value
document_list = dict()  # dictionary of the input documents, key = document id, value = the document



# DO NOT CHANGE THIS METHOD
# Reads the parameters of the project from the parameter file 'file'
# and stores them to the parameter dictionary 'parameters_dictionary'
# accessed as “parameters_dictionary[‘parameter_name’]”
def read_parameters():
    config = configparser.ConfigParser()
    config.read(parameter_file)
    for section in config.sections():
        for key in config[section]:
            if key == 'data':
                parameters_dictionary[key] = config[section][key]
            elif key == 'naive':
                parameters_dictionary[key] = bool(config[section][key])
            elif key == 't':
                parameters_dictionary[key] = float(config[section][key])
            else:
                parameters_dictionary[key] = int(config[section][key])
    

#read_parameters() #remove later

# DO NOT CHANGE THIS METHOD
# Reads all the documents in the 'data_path' and stores them in the dictionary 'document_list'
def read_data(data_path):
    for (root, dirs, file) in os.walk(data_path):
        for f in file:
            file_path = data_path / f
            doc = open(file_path).read().strip().replace('\n', ' ')
            file_id = int(file_path.stem)
            document_list[file_id] = doc


#data_folder = data_main_directory / parameters_dictionary['data']
#read_data(data_folder)
#document_list = {k: document_list[k] for k in sorted(document_list)}

# DO NOT CHANGE THIS METHOD
# Calculates the Jaccard Similarity between two documents represented as sets
def jaccard(doc1, doc2):
    return len(doc1.intersection(doc2)) / float(len(doc1.union(doc2)))


# DO NOT CHANGE THIS METHOD
# Define a function to map a 2D matrix coordinate into a 1D index.
def get_triangle_index(i, j, length):
    if i == j:  # that's an error.
        sys.stderr.write("Can't access triangle matrix with i == j")
        sys.exit(1)
    if j < i:  # just swap the values.
        temp = i
        i = j
        j = temp

    # Calculate the index within the triangular array. Taken from pg. 211 of:
    # http://infolab.stanford.edu/~ullman/mmds/ch6.pdf
    # adapted for a 0-based index.
    k = int(i * (length - (i + 1) / 2.0) + j - i) - 1

    return k


# DO NOT CHANGE THIS METHOD
# Calculates the similarities of all the combinations of documents and returns the similarity triangular matrix
def naive():
    docs_Sets = []  # holds the set of words of each document

    for doc in document_list.values():
        docs_Sets.append(set(doc.split()))

    # Using triangular array to store the similarities, avoiding half size and similarities of i==j
    num_elems = int(len(docs_Sets) * (len(docs_Sets) - 1) / 2)
    similarity_matrix = [0 for x in range(num_elems)]
    for i in range(len(docs_Sets)):
        for j in range(i + 1, len(docs_Sets)):
            similarity_matrix[get_triangle_index(i, j, len(docs_Sets))] = jaccard(docs_Sets[i], docs_Sets[j])

    return similarity_matrix


# METHOD FOR TASK 1
# Creates the k-Shingles of each document and returns a list of them
# Previousy defined a function that removed stop words, but this was not required, and so this doesnt

def k_shingles():
    docs_k_shingles = []
    k = parameters_dictionary['k']

    for _, document in document_list.items():
        
        cleaned_doc = ''.join(c for c in document if c.isalnum() or c.isspace())
        words = cleaned_doc.split()
        k_shingles_set = set(' '.join(words[i:i+k]) 
                             for i in range(len(words) - k + 1))
        docs_k_shingles.append(k_shingles_set)

    return docs_k_shingles



# METHOD FOR TASK 2
# Creates a signatures set of the documents from the k-shingles list
# Creates INPUT MATRIX, name is misleading
def signature_set(k_shingles):

    all_unique_shingles = set().union(*k_shingles)  # can add *k_shingles instead
    all_unique_shingles_list = list(all_unique_shingles)

    shingle_to_index = {shingle: idx for idx,
                        shingle in enumerate(all_unique_shingles_list)}

    num_docs = len(k_shingles)
    num_shingles = len(all_unique_shingles)
    input_matrix = np.zeros((num_shingles, num_docs), dtype=int)

    for doc_idx, shingles_set in enumerate(k_shingles):
        for shingle in shingles_set:
            shingle_idx = shingle_to_index[shingle]
            input_matrix[shingle_idx, doc_idx] = 1

    return input_matrix

# METHOD FOR TASK 3

# Helper to get next prime number
def next_prime(N):
    def is_prime(n):
        if n <= 2:
            return n == 2
        if n % 2 == 0:
            return False
        p = 3
        while p * p <= n:
            if n % p == 0:
                return False
            p += 2
        return True

    prime = N + 1
    while not is_prime(prime):
        prime += 1
    return prime

# A function for generating hash functions
# Returns a dict, where key is permutations and params are a, b, p
def generate_hash_functions(num_perm, N):
    hash_funcs = []
    for i in range(1, num_perm + 1):
        a = random.randint(1, N)
        b = random.randint(0, N)
        p = next_prime(N)
        hash_func = (lambda x, a=a, b=b, p=p: ((a * x + b) %
                     (p)) + 1, {'a': a, 'b': b, 'p': p})
        hash_funcs.append(hash_func)
    return hash_funcs


# Creates the minHash signatures after generating hash functions
'''
less intuitive than pseudocode, but more efficient:
If the shingle is present in the document, iterate through each hash. Compute the hash value, and update it 
to the min the current and the previous.
'''
def minHash(docs_signature_sets, hash_fn):
    
    input_matrix = docs_signature_sets  # for simplicity and readability

    num_shingles = input_matrix.shape[0]  # num rows
    num_docs = input_matrix.shape[1]  # num columns
    num_permutation = len(hash_fn)
    min_hash_signatures = np.full((num_permutation, num_docs), np.inf)

    for shingle in range(num_shingles):  # for each shingle, row
        for doc in range(num_docs):  # for each doc, column
            if input_matrix[shingle, doc] == 1: # ensures that hash functions are applied only to shingles that actually appear in the document
                for permutation, (hash_func, params) in enumerate(hash_fn):
                    shingle_hash = hash_func(shingle, **params)
                    min_hash_signatures[permutation, doc] = min( #permutation is the row, doc is the column if Sig_M
                        min_hash_signatures[permutation, doc], shingle_hash)

    return min_hash_signatures


# METHOD FOR TASK 4
# Hashes the MinHash Signature Matrix into buckets and find candidate similar documents
def lsh(min_hash_signatures):

    b = parameters_dictionary['b']
    num_rows = min_hash_signatures.shape[0] # num permutations
    num_docs = min_hash_signatures.shape[1] 
    rows_per_band = num_rows // b # divide the rows of the signature matrix into b bands
    candidates = set()

    for band in range(b):
        start_index = band * rows_per_band
        end_index = start_index + rows_per_band
        buckets = {}

        for doc in range(num_docs):
            # extract MinHash values for the current document in the current band
            band_slice = tuple(min_hash_signatures[start_index:end_index, doc]) 

            band_hash = hash(band_slice) #fixed size integer

            if band_hash not in buckets: # if band_hash is not already a key in buckets
                buckets[band_hash] = [doc] # initialize a new list with the current doc
            else: # there are documents with identical hash in this band
                for candidate_doc in buckets[band_hash]: 
                    candidates.add((candidate_doc, doc)) # doc becomes candidate with each candidate_doc in the bucket
                buckets[band_hash].append(doc) # need index for later when calculating similarity

    return candidates


# METHOD FOR TASK 5
# Calculates the similarities of the candidate documents
def candidates_similarities(candidate_docs, min_hash_matrix):
    similarity_dict = {}
    t = parameters_dictionary['t']

    for candidate_pair in candidate_docs:

        doc1, doc2 = list(candidate_pair)

        agreement = np.sum(min_hash_matrix[:, doc1] == min_hash_matrix[:, doc2])

        similarity = agreement / min_hash_matrix.shape[0]

        if similarity > t:
            similarity_dict[candidate_pair] = similarity

    return similarity_dict



# DO NOT CHANGE THIS METHOD
# The main method where all code starts
if __name__ == '__main__':
    # Reading the parameters
    read_parameters()

    # Reading the data
    print("\nData reading...")
    data_folder = data_main_directory / parameters_dictionary['data']
    t0 = time.time()
    read_data(data_folder)
    document_list = {k: document_list[k] for k in sorted(document_list)}
    t1 = time.time()
    print(len(document_list), "documents were read in", t1 - t0, "sec\n")

    # Naive
    naive_similarity_matrix = []
    if parameters_dictionary['naive']:
        print("Starting to calculate the similarities of documents...")
        t2 = time.time()
        naive_similarity_matrix = naive()
        t3 = time.time()
        print("Calculating the similarities of", len(naive_similarity_matrix),
              "combinations of documents took", t3 - t2, "sec\n")
    

    # k-Shingles
    print("Starting to create all k-shingles of the documents...")
    t4 = time.time()
    all_docs_k_shingles = k_shingles()
    t5 = time.time()
    print("Representing documents with k-shingles took", t5 - t4, "sec\n")

    # signatures sets (input matrix)
    print("Starting to create the signatures of the documents...")
    t6 = time.time()
    signature_sets = signature_set(all_docs_k_shingles)
    t7 = time.time()
    print("Signatures representation took", t7 - t6, "sec\n")

    # Permutations (real signature matrix)
    print("Starting to simulate the MinHash Signature Matrix...")
    t8 = time.time()
    hash_fn = generate_hash_functions(parameters_dictionary['permutations'], len(signature_sets))
    min_hash_signatures = minHash(signature_sets, hash_fn)
    t9 = time.time()
    print("Simulation of MinHash Signature Matrix took", t9 - t8, "sec\n")

    # LSH
    print("Starting the Locality-Sensitive Hashing...")
    t10 = time.time()
    candidate_docs = lsh(min_hash_signatures)
    t11 = time.time()
    print("LSH took", t11 - t10, "sec\n")

    # Return the over t similar pairs
    print("Starting to get the pairs of documents with over ", parameters_dictionary['t']*100, "% similarity...") # edited to get right percentage
    t14 = time.time()
    true_pairs = candidates_similarities(candidate_docs, min_hash_signatures)
    t15 = time.time()
    print(f"The total number of candidate pairs from LSH: {len(candidate_docs)}")
    print(f"The total number of true pairs from LSH: {len(true_pairs)}")
    print(f"The total number of false positives from LSH: {len(candidate_docs) - len(true_pairs)}")

    if parameters_dictionary['naive']:
        print("Naive similarity calculation took", t3 - t2, "sec")

    print("LSH process took in total", t15 - t14, "sec") # edited to get right order

    '''
    print("The pairs of documents are:\n")
    for p in true_pairs:
        print(f"LSH algorithm reveals that the BBC article {list(p.keys())[0][0]+1}.txt and {list(p.keys())[0][1]+1}.txt \
              are {round(list(p.values())[0],2)*100}% similar")

        print("\n")
    '''
    
    # edited this slightly to make it work with our lsh function
    print("The pairs of documents are:\n")
    for pair, similarity in true_pairs.items():
        doc1, doc2 = pair  # Assuming pair is a tuple (doc1, doc2)
        # multiply by 100 to get percentage
        print(f"LSH algorithm reveals that the BBC article {doc1+1}.txt and {doc2+1}.txt "
            f"are {round(similarity*100, 2)}% similar.\n")
