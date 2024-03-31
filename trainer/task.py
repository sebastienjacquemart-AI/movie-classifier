import argparse

from transformers import Trainer
from datasets import load_from_disk

import trainer.data as data
import trainer.model as model

def train_model(params):
    train_df, test_df = data.create_train_and_test_dataframe("data")

    #dataset = data.create_dataset(train_df, test_df)

    #tokenized_dataset = dataset.map(lambda x: data.preprocess_function(data.get_classes(train_df), x))
    
    #tokenized_dataset.save_to_disk('tokenized_dataset')

    tokenized_dataset = load_from_disk('tokenized_dataset')
    
    if params.explore:
        data.explore_data(train_df, test_df)

    trainer = Trainer(
        model=model.get_solution(data.get_classes(train_df)),
        args=model.get_training_args(),
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=data.get_tokenizer(),	
        data_collator=data.get_data_collator(),
        compute_metrics=model.compute_metrics,
    )

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--explore', action='store_true', help='Whether to explore the data')
    args = parser.parse_args()

    train_model(args)