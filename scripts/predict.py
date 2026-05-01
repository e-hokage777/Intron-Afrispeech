import argparse
import lightning as pl
import pandas as pd
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor

from model import ASRModel
from dataset import PredictDataset, PredictCollator


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_csv", type=str, default="data/test.csv")
    parser.add_argument("--output_csv", type=str, default="submissions/submission.csv")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="gpu")

    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/wav2vec2-base-960h"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    processor = Wav2Vec2Processor.from_pretrained(args.model_name)

    dataset = PredictDataset(
        csv_path=args.test_csv,
        processor=processor
    )

    collator = PredictCollator(processor)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=4
    )

    model = ASRModel.load_from_checkpoint(
        args.checkpoint,
        model_name=args.model_name
    )

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices
    )

    predictions = trainer.predict(model, dataloaders=dataloader)

    # Flatten predictions
    preds = []
    for batch in predictions:
        preds.extend(batch["preds"])

    # Save submission
    df = pd.read_csv(args.test_csv)
    df["transcript"] = preds
    df.to_csv(args.output_csv, index=False)

    print(f"Saved to {args.output_csv}")


if __name__ == "__main__":
    main()