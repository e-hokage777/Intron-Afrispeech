import pandas as pd
from sklearn.model_selection import train_test_split
import os
from core import config as cfg

if __name__ == "__main__":
    df = pd.read_csv("data/train_metadata.csv")
    df["audio_path"] = df["audio_path"].str.split("/").str[2:].str.join("/")

    ## strip out missing data
    indices_to_remove = []
    for index, row in df.iterrows():
        if not os.path.exists(cfg.DATA_BASE_PATH + "/" + row["audio_path"]):
            indices_to_remove.append(index)
    
    df.drop(indices_to_remove, inplace=True)

    train_df, val_df = train_test_split(df, test_size=0.3, random_state=42)

    train_df.to_csv("data/train.csv", index=False)
    val_df.to_csv("data/val.csv", index=False)

    print("Train test splits created")
