import pandas as pd
from sklearn.model_selection import train_test_split
from core.config import config as cfg

if __name__ == "__main__":
    test_df = pd.read_csv("data/test_metadata.csv")
    dev_df = pd.read_csv("data/dev_metadata.csv")

    df = pd.concat([test_df, dev_df], ignore_index=True)
    df["audio_path"] = df["audio_paths"].str.split("/").str[2:].str.join("/")
    df.drop(["audio_paths"], axis=1, inplace=True)

    df.to_csv("data/test.csv", index=False)

    print("Test data prepared")
