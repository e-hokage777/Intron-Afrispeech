import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor

from dataset import ASRDataset, collate_fn
from model import ASRModel

def main():
    MODEL_NAME = "facebook/wav2vec2-base-960h"

    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)

    train_dataset = ASRDataset(
        csv_path="train.csv",
        processor=processor
    )

    val_dataset = ASRDataset(
        csv_path="val.csv",
        processor=processor
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )

    model = ASRModel(model_name=MODEL_NAME)

    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="gpu",
        devices=1,
        precision=16,
        gradient_clip_val=1.0,
        log_every_n_steps=10
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()