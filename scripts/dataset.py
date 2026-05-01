import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
from utils import normalize_text
from core.config import config as cfg


class ASRDataset(Dataset):
    def __init__(
        self,
        csv_path,
        processor,
        sample_rate=16000,
    ):
        self.df = pd.read_csv(csv_path)
        self.processor = processor
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.df)

    # def load_audio(self, path):
    #     waveform, sr = torchaudio.load(path)
    #     if sr != self.sample_rate:
    #         waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
    #     return waveform.squeeze(0)

    def load_audio(self, path):
        waveform, sr = torchaudio.load(path)

        # ✅ Convert to mono (CRITICAL FIX)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        return waveform

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        audio_path = self.base_path + "/" + row["audio_path"]
        audio = self.load_audio(audio_path)
        text = normalize_text(row["transcript"])

        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")

        labels = self.processor.tokenizer(text, return_tensors="pt").input_ids

        return {
            "input_values": inputs.input_values[0],
            "labels": labels[0],
            "text": text,
        }


# def collate_fn(batch):
#     input_values = [x["input_values"] for x in batch]
#     labels = [x["labels"] for x in batch]
#     texts = [x["text"] for x in batch]

#     input_values = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True)
#     labels = torch.nn.utils.rnn.pad_sequence(
#         labels, batch_first=True, padding_value=-100
#     )

#     return {"input_values": input_values, "labels": labels, "text": texts}


class DataCollatorCTC:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        input_values = [x["input_values"] for x in batch]
        labels = [x["labels"] for x in batch]
        texts = [x["text"] for x in batch]

        # Proper audio padding
        inputs = self.processor.pad(
            {"input_values": input_values},
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        # Proper label padding
        labels_batch = self.processor.tokenizer.pad(
            {"input_ids": labels}, padding=True, return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["input_ids"] == self.processor.tokenizer.pad_token_id, -100
        )

        return {
            "input_values": inputs["input_values"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels,
            "text": texts,
        }


import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset


class PredictDataset(Dataset):
    def __init__(
        self,
        csv_path,
        processor,
        sample_rate=16000,
        base_path="/shared/data/afrispeech/afrispeech-test",
    ):
        self.df = pd.read_csv(csv_path)
        self.processor = processor
        self.sample_rate = sample_rate
        self.base_path = base_path

    def load_audio(self, path):
        try:
            waveform, sr = torchaudio.load(path)

            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0)
            else:
                waveform = waveform.squeeze(0)

            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform, sr, self.sample_rate
                )

            return waveform
        except:
            return None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        if row["split"] == "test":
            audio_path = cfg.DATABASE_TEST_PATH + "/" + row["audio_path"]
        else:
            audio_path = cfg.DATABASE_DEV_PATH + "/" + row["audio_path"]

        audio_path = self.base_path + "/" + row["audio_path"]
        audio = self.load_audio(audio_path)

        if audio is None or len(audio) == 0:
            print("MISSING AUDIO", row["audio_path"])
            audio = torch.zeros(16000)  # fallback 1 sec silence

        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")

        return {"input_values": inputs.input_values[0]}


class PredictCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        input_values = [x["input_values"] for x in batch]

        inputs = self.processor.pad(
            {"input_values": input_values},
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        return {
            "input_values": inputs["input_values"],
            "attention_mask": inputs["attention_mask"],
        }
