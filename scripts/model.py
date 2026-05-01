import torch
import lightning as pl
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import jiwer


class ASRModel(pl.LightningModule):
    def __init__(self, model_name="facebook/wav2vec2-base-960h", lr=1e-6):
        super().__init__()
        self.save_hyperparameters()

        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.model.freeze_feature_encoder()

        self.lr = lr
        
        self.val_preds = []
        self.val_refs = []

    def forward(self, input_values, attention_mask=None):
        return self.model(
            input_values=input_values, attention_mask=attention_mask
        ).logits

    def training_step(self, batch, batch_idx):
        outputs = self.model(
            input_values=batch["input_values"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss

        self._log_lr()
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["input_values"], attention_mask=batch["attention_mask"])
        pred_ids = torch.argmax(logits, dim=-1)

        preds = self.processor.batch_decode(pred_ids)
        refs = batch["text"]

        preds = [p.lower().strip() for p in preds]
        refs = [r.lower().strip() for r in refs]

        self.val_preds.extend(preds)
        self.val_refs.extend(refs)
        
    def on_validation_epoch_end(self):
        wer = jiwer.wer(self.val_refs, self.val_preds)
        self.log("val_wer", wer, prog_bar=True)
        self.val_preds.clear()
        self.val_refs.clear()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        logits = self(batch["input_values"], attention_mask=batch["attention_mask"])

        pred_ids = torch.argmax(logits, dim=-1)

        preds = self.processor.batch_decode(pred_ids)

        preds = [p.lower().strip() for p in preds]

        return {"preds": preds, "text": batch.get("text", None)}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs or 10
        )

        return [optimizer], [scheduler]
    
    
    def _log_lr(self):
         # Get the current LR from the optimizer
        opt = self.optimizers()
        lr = opt.param_groups[0]['lr']
        
        # Set prog_bar=True to show it in the terminal
        self.log("lr", lr*10e8, prog_bar=True, on_step=True)
