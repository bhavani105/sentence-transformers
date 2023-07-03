import os
import gzip
import json
from glob import glob
from tqdm import tqdm
import csv
import logging

from sentence_transformers import InputExample
from sentence_transformers import LoggingHandler

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

# def load_nli(nli_dataset_path):

#     label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
#     train_nli_samples = []
#     with gzip.open(nli_dataset_path, 'rt', encoding='utf8') as fIn:
#         reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
#         for row in reader:
#             if row['split'] == 'train':
#                 label_id = label2int[row['label']]
#                 train_nli_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=label_id))
#     return train_nli_samples

def load_pairs_triples(filepath, max=None):
    dataset = []
    with gzip.open(filepath, "rt") as fIn:
        for line in tqdm(fIn):
            texts = json.loads(line)
            assert(len(texts) == 2 or len(texts) == 3)
            dataset.append(InputExample(texts=texts, guid=None))
            if max is not None and len(dataset) == max:
                break
            
    return dataset

def load_sets(filepath):
    pass

def load_datasets(data_config_file, data_dir, max_examples=None, exclude=[]):
    filepaths = []
    with open(data_config_file,'r') as f:
        data_config = json.load(f)
        for line in data_config:
            filepaths.append(os.path.join(data_dir, line['name']))
    
    datasets_pairs = []
    datasets_triples = []
    count = 0
    for filepath in filepaths:
        if filepath in exclude:
            continue
        logging.info(f"Loading  {filepath}")
        dataset = load_pairs_triples(filepath, max_examples)
        count += len(dataset)
        if len(dataset[0].texts) == 3:
            datasets_triples.append(dataset)
            logging.info(f"Num Examples: {len(dataset)} triples total: {count} {filepath}")
        else:
            datasets_pairs.append(dataset)
            logging.info(f"Num Examples: {len(dataset)} pairs total: {count} {filepath}" )
        
    logging.info(f"Num InputExamples {count}")
    return datasets_pairs, datasets_triples
    
                           
# if __name__ == "__main__":
#     data_dir = "/dccstor/colbert-ir/bsiyer/data/all-MiniLM/stackexchange_title_best_voted_answer_jsonl"
#     filepaths = glob(f'{data_dir}/*.jsonl.gz')
    
#     data_dir = "/dccstor/colbert-ir/bsiyer/data/all-MiniLM/embedding-training-data/"
#     data_config_file = "/dccstor/bsiyer6/sentence-transformers/examples/training/all_minilm/data_config_small.json"
#     datasets_pairs, datasets_triples = load_datasets(data_config_file)
    
#     exclude = [
#         # "/dccstor/colbert-ir/bsiyer/data/all-MiniLM/embedding-training-data/stackexchange_duplicate_questions_title-body_title-body.jsonl.gz",
#         # "/dccstor/colbert-ir/bsiyer/data/all-MiniLM/embedding-training-data/stackexchange_duplicate_questions_body_body.jsonl.gz",
#         # "/dccstor/colbert-ir/bsiyer/data/all-MiniLM/embedding-training-data/squad_pairs.jsonl.gz",
#         # "/dccstor/colbert-ir/bsiyer/data/all-MiniLM/embedding-training-data/NQ-train_pairs.jsonl.gz",
#         # "/dccstor/colbert-ir/bsiyer/data/all-MiniLM/embedding-training-data/stackexchange_duplicate_questions_title_title.jsonl.gz",
#         # "/dccstor/colbert-ir/bsiyer/data/all-MiniLM/embedding-training-data/wikihow.jsonl.gz",
#         # "/dccstor/colbert-ir/bsiyer/data/all-MiniLM/embedding-training-data/AllNLI.jsonl.gz",
#         # "/dccstor/colbert-ir/bsiyer/data/all-MiniLM/embedding-training-data/specter_train_triples.jsonl.gz",
#         # "/dccstor/colbert-ir/bsiyer/data/all-MiniLM/embedding-training-data/S2ORC_title_abstract.jsonl.gz"
#     ]
    
    