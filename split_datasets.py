import os
import fnmatch
import pandas as pd
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification
from FlagEmbedding import BGEM3FlagModel
import torch
import json
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

from scripts.fast_detect_gpt import get_sampling_discrepancy_analytic
from scripts.lastde_doubleplus import get_sampling_discrepancy_lastde
from scripts.model import load_tokenizer, load_model

# Set up argument parser
parser = argparse.ArgumentParser(description='Split datasets and calculate embeddings, cond, and crit.')
parser.add_argument('--device', type=str, default='cuda', help='Device to use for computation')
parser.add_argument('--input_directory', type=str, required=True, help='Input directory path')
parser.add_argument('--output_directory', type=str, required=True, help='Output directory path')
parser.add_argument('--embedding_type', type=str, default='encode', choices=['encode', 'last_hidden_state', 'cls'], help='Type of embedding to use')
parser.add_argument('--embedding_model_name', type=str, default='BAAI/bge-m3',
                   choices=['BAAI/bge-m3', 'roberta-base'],  # Specify optional models
                   help='Model name for embeddings (BAAI/bge-m3 or roberta-base)')
parser.add_argument('--scoring_model_name', type=str, default='EleutherAI/gpt-neo-2.7B', help='Model name for scoring')
parser.add_argument('--reference_model_name', type=str, default='EleutherAI/gpt-neo-2.7B', help='Model name for reference model')
parser.add_argument('--batch_size', type=int, default=500, help='Batch size for processing data')
parser.add_argument('--dataset', type=str, default="xsum")  # for fast-detect load model
parser.add_argument('--cache_dir', type=str, default="../cache")

parser.add_argument('--roberta_model_name', type=str, default='roberta-base-openai-detector', choices=['roberta-large-openai-detector', 'roberta-base-openai-detector'], help='Model name for roberta model')
parser.add_argument('--crit_type', type=str, default='fast', choices=['fast', 'roberta', 'lastde'], help='Type of criterion to use for detection')

# Additional arguments for crit_type "lastde", Using default values from the original code
parser.add_argument('--embed_size', type=int, default=4)
parser.add_argument('--epsilon', type=float, default=8)
parser.add_argument('--tau_prime', type=int, default=15)
parser.add_argument('--n_samples', type=int, default=100)

args = parser.parse_args()

# If reference model is not specified, use scoring model as reference
if args.reference_model_name is None:
    args.reference_model_name = args.scoring_model_name

# Set device
device = torch.device("cuda")

# Initialize embedding model and tokenizer
if args.embedding_type == 'encode':
    embedding_tokenizer = AutoTokenizer.from_pretrained(args.embedding_model_name)
    embedding_model = BGEM3FlagModel(args.embedding_model_name, device=device)
else:
    embedding_tokenizer = AutoTokenizer.from_pretrained(args.embedding_model_name)
    embedding_model = AutoModel.from_pretrained(args.embedding_model_name)
    embedding_model.to(device)

scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.dataset, args.cache_dir)
scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)
scoring_model.eval()

def from_pretrained(cls, model_name, kwargs, cache_dir):
    # use local model if it exists
    local_path = os.path.join(cache_dir, 'local.' + model_name.replace("/", "_"))
    if os.path.exists(local_path):
        return cls.from_pretrained(local_path, **kwargs)
    return cls.from_pretrained(model_name, **kwargs, cache_dir=cache_dir)

if args.crit_type == "fast" or args.crit_type == "lastde":
    if args.reference_model_name != args.scoring_model_name:
        reference_tokenizer = load_tokenizer(args.reference_model_name, args.dataset, args.cache_dir)
        reference_model = load_model(args.reference_model_name, args.device, args.cache_dir)
        reference_model.eval()

    # Initialize reference model and tokenizer if different from scoring model
    if args.reference_model_name != args.scoring_model_name:
        reference_tokenizer = AutoTokenizer.from_pretrained(args.reference_model_name)
        reference_model = AutoModelForCausalLM.from_pretrained(args.reference_model_name)
        reference_model.to(device)
        reference_model.eval()
    else:
        reference_tokenizer = scoring_tokenizer
        reference_model = scoring_model
elif args.crit_type == "roberta":
    roberta_detector = from_pretrained(AutoModelForSequenceClassification, args.roberta_model_name, {}, args.cache_dir).to(device)
    roberta_tokenizer = from_pretrained(AutoTokenizer, args.roberta_model_name, {}, args.cache_dir)
    roberta_detector.eval()
else:
    raise ValueError(f"Unknown crit type: {args.crit_type}")

def get_embedding(text, embedding_type="last_hidden_state"):
    """
    Get embeddings for input text
    """
    inputs = embedding_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    if embedding_type == 'encode':
        with torch.no_grad():
            embedding = embedding_model.encode(text)['dense_vecs']  # (1024,)
    elif embedding_type == 'last_hidden_state':
        with torch.no_grad():
            outputs = embedding_model(**inputs)
        embedding = outputs.last_hidden_state.squeeze().cpu().numpy()  # (282, 768)
    elif embedding_type == 'cls':
        with torch.no_grad():
            outputs = embedding_model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy() # (768, )
    return embedding

def calculate_ngram_repetition(text, n=2):
    """
    Calculate the n-gram repetition rate of the text
    :param text: Input text
    :param n: The n value for n-gram
    :return: n-gram repetition rate
    """
    words = text.split()
    ngrams = list(zip(*[words[i:] for i in range(n)]))  # Generate n-grams
    ngram_counts = Counter(ngrams)
    total_ngrams = len(ngrams)
    repeated_ngrams = sum(count for count in ngram_counts.values() if count > 1)
    return repeated_ngrams / total_ngrams if total_ngrams > 0 else 0

def calculate_tfidf(text):
    """
    Calculate the TF-IDF value of the text
    :param text: Input text, as corpus: text corpus
    :param corpus: Text corpus
    :return: The average TF-IDF value of the text
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(text)
    tfidf_scores = tfidf_matrix[-1].toarray()[0]  # Get the TF-IDF vector of the input text
    return tfidf_scores.mean()

def calculate_ttr(text):
    """
    Calculate the corrected TTR (Type-Token Ratio)
    :param text: Input text
    :return: Corrected TTR value
    """    
    words = text.split()
    unique_tokens = len(set(words))
    total_tokens = len(words)
    return unique_tokens / (total_tokens ** 0.5) if total_tokens > 0 else 0

def get_cond_and_crit(text):
    """
    计算 cond 和 crit 值，包括 n-gram 重复率和 TF-IDF
    :param text: 输入文本
    :param corpus: 文本语料库（用于 TF-IDF 计算）
    :return: cond 和 crit
    """
    # Calculate cond
    tokenized = scoring_tokenizer(text, truncation=True, return_tensors="pt", padding=True, return_token_type_ids=False).to(device)
    labels = tokenized.input_ids[:, 1:]

    if args.crit_type == "lastde":
        criterion_fn = get_sampling_discrepancy_lastde
    else:
        criterion_fn = get_sampling_discrepancy_analytic
    
    with torch.no_grad():
        logits_score = scoring_model(**tokenized).logits[:, :-1]
        
    # Calculate cond
    cond = [
        logits_score.size(1),  # sequence length
        logits_score.mean().item(),  # mean logit score
        logits_score.var().item()  # variance of logit score
    ]

    # Add n-gram repetition rate
    ngram_repetition = calculate_ngram_repetition(text, n=2)
    cond.append(ngram_repetition)
    ngram_repetition = calculate_ngram_repetition(text, n=3)
    cond.append(ngram_repetition)

    # Add corrected TTR
    ttr = calculate_ttr(text)
    cond.append(ttr)

    # Calculate crit
    if args.crit_type == "fast":
        with torch.no_grad():
            if args.reference_model_name == args.scoring_model_name:
                logits_ref = logits_score  # ref is sample q, score is p
            else:
                ref_tokenized = reference_tokenizer(text, truncation=True, return_tensors="pt", padding=True, return_token_type_ids=False).to(device)
                assert torch.all(ref_tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = reference_model(**ref_tokenized).logits[:, :-1]
            
            crit = criterion_fn(logits_ref, logits_score, labels)
    elif args.crit_type == "roberta":
        roberta_tokenized = roberta_tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            crit = roberta_detector(**roberta_tokenized).logits.softmax(-1)[0, 0].item()

    elif args.crit_type == "lastde":
        with torch.no_grad():
            if args.reference_model_name == args.scoring_model_name:
                logits_ref = logits_score  # ref is sample q, score is p
            else:
                ref_tokenized = reference_tokenizer(text, truncation=True, return_tensors="pt", padding=True, return_token_type_ids=False).to(device)
                assert torch.all(ref_tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = reference_model(**ref_tokenized).logits[:, :-1]
            
            crit = criterion_fn(logits_ref, logits_score, labels, args)
    else:
        raise ValueError(f"Unknown crit type: {args.crit_type}")
    
    return cond, crit

def distribute_datasets(input_dir, output_dir, batch_size=500):
    """
    Process datasets, calculate embeddings, cond, and crit, and distribute to output files
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize a dictionary to store grouped data
    datasets = {}
    # Track processed rows
    processed_rows = 0
    
    # Iterate through all files in input directory
    for filename in os.listdir(input_dir):
        if fnmatch.fnmatch(filename, '*.csv'):
            filepath = os.path.join(input_dir, filename)
            print(f"Processing file: {filename}")
            
            # Read CSV file
            try:
                df = pd.read_csv(filepath)
            except Exception as e:
                print(f"Cannot read file {filename}: {e}")
                continue
            
            # Process each row
            for index, row in df.iterrows():
                # Convert Pandas Series to dictionary
                row_dict = row.to_dict()
                
                # Extract dataset name from src column
                src_value = row_dict.get('src', None)
                if not src_value:
                    print(f"Warning: Row {index} in file {filename} has no src column, skipping.")
                    continue
                
                # Split string by first underscore
                first_dash_index = src_value.find('_')
                if first_dash_index == -1:
                    dataset_name = src_value.lower()
                else:
                    dataset_name = src_value[:first_dash_index].lower()

                # Extract text and calculate embeddings
                text = row_dict.get('text', None)
                if text:
                    # Get embedding
                    embedding = get_embedding(text, embedding_type=args.embedding_type)
                    row_dict['embedding'] = embedding.tolist()  # Convert NumPy array to list
                    
                    # Get cond and crit
                    try:
                        cond, crit = get_cond_and_crit(text)
                        row_dict['cond'] = cond  # Add cond to row data
                        row_dict['crit'] = crit  # Add crit to row data
                    except Exception as e:
                        print(f"Error calculating cond/crit for row {index} in file {filename}: {e}")
                        continue
                else:
                    print(f"Warning: Row {index} in file {filename} has no text column, skipping.")
                    continue
                
                # Add data to corresponding group
                if dataset_name not in datasets:
                    datasets[dataset_name] = []
                datasets[dataset_name].append(row_dict)
                
                # Save and clear memory after processing batch_size rows
                processed_rows += 1
                if processed_rows >= batch_size:
                    # Save each group's data to corresponding file
                    for name, data in datasets.items():
                        output_filename = os.path.join(output_dir, f"{name}_dataset.json")
                        # If file exists, try to append data; otherwise create new file
                        if os.path.exists(output_filename):
                            try:
                                with open(output_filename, 'r') as f:
                                    existing_data = json.load(f)
                            except json.JSONDecodeError:
                                # If file content is not valid JSON, initialize as empty list
                                existing_data = []
                            existing_data.extend(data)
                            with open(output_filename, 'w') as f:
                                json.dump(existing_data, f, indent=4)
                        else:
                            with open(output_filename, 'w') as f:
                                json.dump(data, f, indent=4)
                        print(f"Dataset {name} has been saved to {output_filename}")
                    
                    # Clear datasets dictionary and processed_rows counter
                    datasets = {}
                    processed_rows = 0
                    print("Memory cleared, continuing processing...")
            
            print(f"Finished processing {filename}")
    
    # After processing all files, save remaining data
    for name, data in datasets.items():
        output_filename = os.path.join(output_dir, f"{name}_dataset.json")
        # If file exists, try to append data; otherwise create new file
        if os.path.exists(output_filename):
            try:
                with open(output_filename, 'r') as f:
                    existing_data = json.load(f)
            except json.JSONDecodeError:
                # If file content is not valid JSON, initialize as empty list
                existing_data = []
            existing_data.extend(data)
            with open(output_filename, 'w') as f:
                json.dump(existing_data, f, indent=4)
        else:
            with open(output_filename, 'w') as f:
                json.dump(data, f, indent=4)
        print(f"Dataset {name} has been saved to {output_filename}")


if __name__ == "__main__":
    distribute_datasets(args.input_directory, args.output_directory, args.batch_size)