import torch
import lightning as pl
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import jiwer

class ASRModel(pl.LightningModule):
    def __init__(self, model_name="facebook/wav2vec2-base-960h", lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)

        self.lr = lr

    def forward(self, input_values):
        return self.model(input_values).logits

    def training_step(self, batch, batch_idx):
        outputs = self.model(
            input_values=batch["input_values"],
            labels=batch["labels"]
        )
        loss = outputs.loss

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["input_values"])
        pred_ids = torch.argmax(logits, dim=-1)

        preds = self.processor.batch_decode(pred_ids)
        refs = batch["text"]

        preds = [p.lower().strip() for p in preds]
        refs = [r.lower().strip() for r in refs]

        wer = jiwer.wer(refs, preds)

        self.log("val_wer", wer, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10
        )

        return [optimizer], [scheduler]