import trainer.data as data
import argparse

def train_model(params):
    train_df, test_df = data.create_train_and_test_dataframe("data")
    train_df = data.preprocess_labels(train_df)

    data.explore_data(train_df, test_df)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    train_model(args)