import os
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding

import trainer.boiler as boiler

def create_train_and_test_dataframe(data_path):
    train_file = os.path.join(data_path, "train.csv")
    test_file = os.path.join(data_path, "test2.csv")

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file, delimiter=';')

    return train_df, test_df

def explore_data(train_df, test_df):
    # Check columns
    print(f"There are {train_df.columns} columns in train set.")
    print(f"There are {test_df.columns} columns in test set.")

    # Check example of data
    print(train_df.head())
    print(test_df.head())

    # Check the number of rows
    print(f"There are {len(train_df)} rows in the train set.")
    print(f"There are {len(test_df)} rows in the test set.")
    print(f"Ratio train to test: {len(train_df)/(len(train_df)+len(test_df))}.")
    
    # Check the number of duplicate rows
    total_duplicate_syn_train = sum(train_df["synopsis"].duplicated())
    print(f"There are {total_duplicate_syn_train} duplicate synopsis in train set.")
    total_duplicate_syn_test = sum(test_df["synopsis"].duplicated())
    print(f"There are {total_duplicate_syn_test} duplicate synopsis in test set.")

    #total_duplicate_syn_and_tag_train = sum(train_df.duplicated(subset=["synopsis", "tags"]))

    # Plot the sum of occurrences for each label
    x = train_df.iloc[:, 3:].sum()
    boiler.plot_graph(x, "Class counts", "Label", "# of Occurrences")

    # Plot the number of labels per synopsis


    return

def preprocess_function(classes, example):
    text = example['synopsis']
    if example["tags"] is not None:
        all_labels = example['tags'].split(', ')
    else: 
        all_labels = []
    labels = [0. for i in range(len(classes))]
    for label in all_labels:
        label_id = get_class2id(classes)[label]
        labels[label_id] = 1.

    example = get_tokenizer()(text, truncation=True)
    example['labels'] = labels
    
    return example

def create_dataset(train_df, test_df):
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    dataset = DatasetDict()

    dataset['train'] = train_dataset
    dataset['test'] = test_dataset

    return dataset

def get_tokenizer(model_path="microsoft/deberta-v3-small"):
    return AutoTokenizer.from_pretrained(model_path)

def get_data_collator():
    return DataCollatorWithPadding(tokenizer=get_tokenizer())

def get_classes(df):
    all_tags = ','.join(df["tags"].astype(str))

    individual_tags = all_tags.split(",")
    
    unique_classes = set([tag.strip() for tag in individual_tags])
    
    return list(unique_classes)

def get_id2class(classes):
    id2class = {id:class_ for class_, id in enumerate(classes)}
    return id2class

def get_class2id(classes):
    class2id = {class_:id for id, class_ in enumerate(classes)}
    return class2id
