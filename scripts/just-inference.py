import argparse
import torch
import torchaudio
import pandas as pd
from tqdm import tqdm
from core.config import config as cfg


# -----------------------------
# Load pipeline (from tutorial)
# -----------------------------
def load_pipeline(device):
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device)
    labels = bundle.get_labels()
    sample_rate = bundle.sample_rate

    return model, labels, sample_rate


# -----------------------------
# Greedy decoder (tutorial style)
# -----------------------------
def greedy_decode(emission, labels):
    indices = torch.argmax(emission, dim=-1)

    # Collapse repeats + remove blanks (index 0)
    indices = torch.unique_consecutive(indices, dim=-1)

    tokens = []
    for i in indices:
        if i != 0:
            tokens.append(labels[i])

    return "".join(tokens).replace("|", " ").strip().lower()


# -----------------------------
# Audio loader
# -----------------------------
def load_audio(path, target_sr):
    waveform, sr = torchaudio.load(path)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    else:
        waveform = waveform.squeeze(0)

    # Resample
    if sr != target_sr:
        waveform = torchaudio.functional.resample(
            waveform, sr, target_sr
        )

    return waveform


# -----------------------------
# Inference
# -----------------------------
def transcribe(model, labels, waveform, device):
    waveform = waveform.to(device)

    with torch.no_grad():
        emission, _ = model(waveform.unsqueeze(0))  # (1, T, vocab)

    emission = emission[0].cpu()

    return greedy_decode(emission, labels)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_csv", type=str, default="data/test.csv")
    parser.add_argument("--output_csv", type=str, default="submission-ji.csv")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load pipeline
    model, labels, sample_rate = load_pipeline(device)
    model.eval()

    # Load test data
    df = pd.read_csv(args.test_csv)

    predictions = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        if row["split"] == "test":
            audio_path = cfg.DATABASE_TEST_PATH + "/" + row["audio_path"]
        else:
            audio_path = cfg.DATABASE_DEV_PATH + "/" + row["audio_path"]

        try:
            waveform = load_audio(audio_path, sample_rate)
        except Exception as e:
            print(f"[ERROR] {audio_path}: {e}")
            predictions.append("")
            continue

        if waveform is None or len(waveform) == 0:
            predictions.append("")
            continue

        text = transcribe(model, labels, waveform, device)
        predictions.append(text)

    # Save submission
    df["transcript"] = predictions
    df.to_csv(args.output_csv, index=False)

    print(f"Saved predictions to {args.output_csv}")


if __name__ == "__main__":
    main()