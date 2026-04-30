import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
from utils import normalize_text

class ASRDataset(Dataset):
    def __init__(self, csv_path, processor, sample_rate=16000):
        self.df = pd.read_csv(csv_path)
        self.processor = processor
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.df)

    def load_audio(self, path):
        waveform, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        return waveform.squeeze(0)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        audio = self.load_audio(row["audio_path"])
        text = normalize_text(row["transcript"])

        inputs = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        with self.processor.as_target_processor():
            labels = self.processor(text, return_tensors="pt").input_ids

        return {
            "input_values": inputs.input_values[0],
            "labels": labels[0],
            "text": text
        }


def collate_fn(batch):
    input_values = [x["input_values"] for x in batch]
    labels = [x["labels"] for x in batch]
    texts = [x["text"] for x in batch]

    input_values = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "input_values": input_values,
        "labels": labels,
        "text": texts
    }