import os
import pandas as pd

import trainer.boiler as boiler

def create_train_and_test_dataframe(data_path):
    train_file = os.path.join(data_path, "train.csv")
    test_file = os.path.join(data_path, "test.csv")

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

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

# @HuggingFace: Could I make package of this???
def preprocess_labels(df):
    # Split tags into separate columns
    tags_split = df['tags'].str.get_dummies(sep=', ')

    # Merge the new columns with the original DataFrame
    df = pd.concat([df, tags_split], axis=1)

    # Drop the original 'tags' column if needed
    # df.drop('tags', axis=1, inplace=True)

    return df
