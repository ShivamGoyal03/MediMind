import pandas as pd
from datasets import load_dataset
from src.utils.helpers import clean_text

def load_raw_data():
    """
    Loads the MedQuad dataset from Hugging Face.

    Returns:
        tuple: Training and testing DataFrames.
    """
    dataset = load_dataset("Amirkid/MedQuad-dataset")
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])
    return train_df, test_df

def clean_data(df: pd.DataFrame, text_columns: list) -> pd.DataFrame:
    """
    Cleans specified text columns in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to clean.
        text_columns (list): List of column names containing text to clean.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    for column in text_columns:
        if column in df.columns:
            df[column] = df[column].apply(lambda x: clean_text(x))
    return df

def preprocess_and_save():
    """
    Loads, cleans, and saves the processed data.
    """
    print("Loading raw data...")
    train_df, test_df = load_raw_data()
    
    # Specify which columns to clean
    text_columns = ['question', 'context', 'answer']
    
    print("Cleaning training data...")
    train_df = clean_data(train_df, text_columns)
    
    print("Cleaning testing data...")
    test_df = clean_data(test_df, text_columns)
    
    print("Saving processed data to CSV files...")
    train_df.to_csv('data/processed/train.csv', index=False)
    test_df.to_csv('data/processed/test.csv', index=False)
    
    print("Data preprocessing completed successfully.")

if __name__ == "__main__":
    preprocess_and_save()