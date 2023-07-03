import sys
import os
import gzip
import csv
import random
import json
import logging
import dataclasses
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from datetime import datetime
from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from MultiDatasetDataLoader import MultiDatasetDataLoader
from load_datasets import load_datasets


#### print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
 
@dataclass
class ProcessArguments:
    model_name_or_path: str = field(
        default="nreimers/MiniLM-L6-H384-uncased",
        metadata={"help": "model name or path"}
    )
    output_dir: str = field(
        default='output', metadata={"help": "output directory"}
    )
    data_dir: str = field(
        default='embedding-training-data', metadata={"help": "directory containing the data"}
    )
    data_config_file: str = field(
        default='./config/data_config_5M.json', metadata={"help": "data config file"}
    )
    use_data_weights_from_config: bool = field(
        default=False, metadata={"help": "Use data weights from data_config_file"}
    )
    eval_dataset_path: str = field(
        default="stsbenchmark/stsbenchmark.tsv.gz", metadata={"help": "data config file"}
    )
    num_epochs: int = field(
        default=1, metadata={"help": "num epochs"}
    )
    batch_size_pairs: int = field(
        default=384, metadata={"help": "batch size for examples as pairs"}
    )
    batch_size_triplets: int = field(
        default=256, metadata={"help": "batch size for examples as triplets"}
    )
    max_seq_length: int = field(
        default=512, metadata={"help": "max seq length"}
    )
    evaluation_steps: int = field(
        default=1000, metadata={"help": "eval steps"}
    )
    checkpoint_save_steps: int = field(
        default=3, metadata={"help": "steps to checkpoint"}
    )
    checkpoint_save_limit: int = field(
        default=1000, metadata={"help": "max checkpoints to save"}
    )
    warmup_steps: int = field(
        default=500, metadata={"help": "warm up steps"}
    )
    steps_per_epoch: int = field(
        default=7800, metadata={"help": "steps per epoch"}
    )
    max_train_examples: int = field(
        default=None, metadata={"help": "max number of examples to load per dataset"}
    )
    use_amp: bool = field(
        default=True, metadata={"help": "Set to False, if you use a CPU or your GPU does not support FP16 operations"}
    )
    learning_rate: float = field(
        default=2e-5, metadata={"help": "learning rate"}
    )

def main():
    parser = HfArgumentParser(ProcessArguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        args = parser.parse_args_into_dataclasses()[0]
        
    # dump args
    logging.info(json.dumps(dataclasses.asdict(args),indent=2))
    
    
    random.seed(1234)

    model_name = args.model_name_or_path
    num_epochs = args.num_epochs
    eval_data_path = args.eval_dataset_path 
    batch_size_pairs = args.batch_size_pairs
    batch_size_triplets = args.batch_size_triplets
    max_seq_length = args.max_seq_length
    use_amp = args.use_amp                 
    evaluation_steps = args.evaluation_steps
    checkpoint_save_steps = args.checkpoint_save_steps
    warmup_steps = args.warmup_steps
    max_train_examples = args.max_train_examples
    steps_per_epoch = args.steps_per_epoch
    learning_rate = args.learning_rate
    checkpoint_save_limit = args.checkpoint_save_limit

    # Save path of the model
    model_save_path = f"{args.output_dir}_ep{num_epochs}_steps{steps_per_epoch}_lr{learning_rate}"

    ## Setup SentenceTransformer for training
    ## TODO: add option to normalize embeddings 
    ## TODO: weighted mean pooling - idf / colbert max sim ??
    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    ## Read in the data
    data_dir = args.data_dir
    data_config_file = args.data_config_file
    datasets_pairs, datasets_triples = load_datasets(data_config_file, data_dir=data_dir, max_examples=max_train_examples)
    
    
    if steps_per_epoch is None or steps_per_epoch == 0:
        steps_per_epoch = max([len(data) for data in datasets_pairs]) / batch_size_pairs
        logging.info(f"Steps per epoch: {steps_per_epoch}")

    # Setup dataloader and loss pairs
    # TODO: use custom data weights
    # TODO: support in batch negatives for triplets 
    # TODO: ColBERT / DPR style approach to in-batch negatives
    # TODO: enable streaming data using HF datasets
    train_objectives = []
    if len(datasets_pairs) > 0:
        train_dataloader_pairs = MultiDatasetDataLoader(datasets_pairs, batch_size_pairs=batch_size_pairs, dataset_size_temp=1, allow_swap=False)
        train_loss_pairs = losses.MultipleNegativesRankingLoss(model, scale = 20.0, similarity_fct = util.cos_sim)
        train_objectives.append( (train_dataloader_pairs, train_loss_pairs) )
    if len(datasets_triples) > 0:
        train_dataloader_triples = MultiDatasetDataLoader(datasets_triples, batch_size_pairs=batch_size_pairs, batch_size_triplets=batch_size_triplets, dataset_size_temp=1, allow_swap=False)
        train_loss_triples = losses.TripletLoss(model, distance_metric=losses.TripletDistanceMetric.EUCLIDEAN, triplet_margin=5)
        train_objectives.append( (train_dataloader_triples, train_loss_triples) )
        
    #Read STSbenchmark dataset and use it as development set
    logging.info("Read STSbenchmark dev dataset")
    dev_samples = []
    with gzip.open(eval_data_path, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            if row['split'] == 'dev':
                score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
                dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

    dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')

    # Configure the training
    logging.info("Warmup-steps: {}".format(warmup_steps))
    
    optimizer_params = { 'lr': learning_rate }

    # Train the model
    # TODO: migrate to HF style trainer
    model.fit(train_objectives=train_objectives,
            evaluator=dev_evaluator,
            epochs=num_epochs,
            evaluation_steps=evaluation_steps,
            warmup_steps=warmup_steps,
            steps_per_epoch=steps_per_epoch,
            output_path=model_save_path,
            use_amp=use_amp,
            checkpoint_path=model_save_path,
            checkpoint_save_total_limit=checkpoint_save_limit,
            optimizer_params=optimizer_params
            )


if __name__ == "__main__":
    main()