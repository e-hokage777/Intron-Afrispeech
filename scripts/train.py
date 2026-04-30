import argparse
import lightning as pl
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor
from lightning.pytorch.callbacks import ModelCheckpoint

from dataset import ASRDataset, collate_fn
from model import ASRModel


def parse_args():
    parser = argparse.ArgumentParser()

    # Training params
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)

    # Hardware
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", type=int, default=16)

    # Trainer tweaks
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)

    # Paths
    parser.add_argument("--train_csv", type=str, default="train.csv")
    parser.add_argument("--val_csv", type=str, default="val.csv")

    ## Data loading
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pin_memory", type=bool, default=True)

    # Model
    parser.add_argument("--model_name", type=str, default="facebook/wav2vec2-base-960h")

    return parser.parse_args()


def main():
    args = parse_args()

    processor = Wav2Vec2Processor.from_pretrained(args.model_name)

    train_dataset = ASRDataset(csv_path=args.train_csv, processor=processor)

    val_dataset = ASRDataset(csv_path=args.val_csv, processor=processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    model = ASRModel(model_name=args.model_name, lr=args.lr)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        log_every_n_steps=args.log_every_n_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[
            ModelCheckpoint(
                monitor="val_wer",
                mode="min",
                save_top_k=1,
                dirpath="checkpoints/facebook-wav2vec2-base-960h",
                filename="wav2vec2-base-960h-{epoch:02d}-{val_wer:.2f}",
            )
        ],
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
