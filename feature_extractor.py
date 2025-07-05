import os
import re
import pickle
import mmh3
import numpy as np
from joblib import Parallel, delayed
from nltk.util import ngrams
from opcode_dictionary import move_instructions, branch_instructions, getter_setter_instructions, method_invoke_instructions, logic_arithmetic_instructions
import csv

def extract_opcodes(smali_file):
    opcodes = []
    with open(smali_file, "r", encoding="utf-8") as f:
        for line in f:
            opcode_match = re.search(r'\s+([A-Za-z0-9_/-]+)', line)
            if opcode_match:
                opcode = opcode_match.group(1)
                if opcode in (move_instructions | branch_instructions | getter_setter_instructions | method_invoke_instructions | logic_arithmetic_instructions):
                    opcodes.append(opcode)
    return opcodes

def get_ngrams(directory, n):
    opcodes = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".smali"):
                smali_file_path = os.path.join(root, file)
                opcodes.extend(extract_opcodes(smali_file_path))
    return list(ngrams(opcodes, n)) 
              
def extract_features(sha, directory_path, hash_dim, n):
    if sha.endswith("-decompressed"):  
        smali_file = os.path.join(directory_path, sha)
        tokens = get_ngrams(smali_file, n)
        token_hash_buckets = [(mmh3.hash(' '.join(w)) % (hash_dim - 1) + 1) for w in tokens]
        token_buckets_counts = np.zeros(hash_dim)
        buckets, counts = np.unique(token_hash_buckets, return_counts=True)
        for bucket, count in zip(buckets, counts):
            token_buckets_counts[bucket] = count
        if "malware" in directory_path:
            malware = 1
        else:
            malware = 0          
        return (sha, token_buckets_counts/np.sum(token_buckets_counts), malware)
        
def extract_features_test(directory_path, hash_dim, n):
    smali_file = directory_path
    tokens = get_ngrams(smali_file, n)
    token_hash_buckets = [(mmh3.hash(' '.join(w)) % (hash_dim - 1) + 1) for w in tokens]
    token_buckets_counts = np.zeros(hash_dim)
    buckets, counts = np.unique(token_hash_buckets, return_counts=True)
    for bucket, count in zip(buckets, counts):
        token_buckets_counts[bucket] = count        
    return (token_buckets_counts/np.sum(token_buckets_counts))        

def extract_features_wrapper(file_path, directory_path, hash_dim, n):
    return extract_features(file_path, directory_path, hash_dim, n)
    
def data_preprocessor(benign_files, malicious_files, path_to_benign_files, path_to_malicious_files, features_length, n):
    print(f"Preprocessor working - n = {n}, Buckets = {features_length}")     
    ben_features = Parallel(n_jobs=-1)(
        delayed(extract_features_wrapper)(sha, directory_path=path_to_benign_files, hash_dim=features_length, n=n)
        for sha in benign_files
    )
    print("Benign files processed")
    print(f"Samples: {len(ben_features)} Buckets: {len(ben_features[0][1])}")
    mal_features = Parallel(n_jobs=-1)(
        delayed(extract_features_wrapper)(sha, directory_path=path_to_malicious_files, hash_dim=features_length, n=n)
        for sha in malicious_files
    ) 
    print("Malicious files processed")
    print(f"Samples: {len(mal_features)} Buckets: {len(mal_features[0][1])}")
    all_features = ben_features + mal_features
    processed_data = [(sha, *counts, malware) for sha, counts, malware in all_features]
    print("Data processed.")
    return processed_data

def export_data(data, output_file, features_length):
    with open(output_file, 'w', newline='') as csvfile:
        label_row = "File Name",*[f"Bucket {i}" for i in range(features_length)] ,"Class"
        wr = csv.writer(csvfile)       
        wr.writerow(label_row)
        for sample in data:
            wr.writerow(sample)
    
            
def main():            
    features_length = 2048
    n = 4
    path_to_training_benign_files = ""
    path_to_training_malicious_files = ""
    train_benign_files = os.listdir(path_to_training_benign_files)
    train_malicious_files = os.listdir(path_to_training_malicious_files)

    data = data_preprocessor(
        benign_files=train_benign_files,
        malicious_files=train_malicious_files,
        path_to_benign_files=path_to_training_benign_files,
        path_to_malicious_files=path_to_training_malicious_files,
        features_length=features_length,
        n=n
    )

    output_file = f"{n} {features_length}.csv"
    export_data(data, output_file, features_length)
    output_file = f"{n} {features_length}.pickle"
    
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    print("Finished!")
    
if __name__ == '__main__':
    main()