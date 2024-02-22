import torch
import glob
import os
import librosa
import numpy as np
from typing import Any
from transformers import SeamlessM4TTokenizer, SeamlessM4TAudioModel, AdamW

class DataLoader:
    def __init__(self, batch_size: int, audio_folder: str, transcripts_file: str):
        self.batch_size = batch_size
        self.audio_files = sorted(glob.glob(os.path.join(audio_folder, "*")))
        self.transcripts = []
        with open(transcripts_file, "r") as fp:
            lines = fp.readlines()
            for line in lines:
                self.transcripts.append(line.strip())
        self.current_idx = 0

    def __len__(self) -> int:
        return len(self.transcripts) // self.batch_size

    def __iter__(self) -> Any:
        while True:
            idxs = list(range(self.current_idx, min((self.current_idx + self.batch_size), len(self.transcripts))))
            current_utterances = self.transcripts[idxs]
            current_audios = [librosa.load(f, sr=16000)[0].astype(np.float32) for f in self.audio_files[idxs]]
            max_length = max([len(_) for _ in current_audios])
            padded_audios = [np.pad(_, (max_length - len(_), 0), "constant") for _ in current_audios]
            tensor_audios = torch.FloatTensor(padded_audios)
            self.current_idx += self.batch_size
            yield tensor_audios, tokenizer.encode(current_utterances)

class SeamlessModel:
    def __init__(self, seamless_ckpt_path: str = './seamless_ckpt.tar', device: str = 'cuda'):
        """Loads the Seamless model from the saved checkpoint."""
        checkpoint = torch.load(seamless_ckpt_path, map_location=torch.device(device))
        self.model = checkpoint['state_dict']
        self.model = {k.replace('module.', ''): v for k, v in self.model.items()}
        self.model = nn.Sequential(OrderedDict(self.model))
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs forward pass."""
        x = x.to(self.device)
        output = self.model(x)
        return output

    @staticmethod
    def label_smoothing_crossentropy(logits: torch.Tensor, labels: torch.Tensor, epsilon: float = 0.1) -> torch.Tensor:
        """Custom Label Smoothing Cross Entropy Loss Calculation."""
        n_classes = logits.shape[-1]
        log_probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
        # Calculate one-hot encoding and average probabilities
        one_hot = torch.zeros_like(log_probabilities).scatter_(dim=-1, index=labels[:, None], value=1)
        avg_prob = one_hot * (1 - epsilon) + (1 - one_hot) * epsilon / (n_classes - 1)
        loss = (-avg_prob * log_probabilities).sum(dim=-1)
        return loss.mean()

    def train_loop(self, model: SeamlessM4TAudioModel, dataloader: DataLoader, num_epochs: int, learning_rate: float):
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            print(f"Starting epoch {epoch}")
            total_loss = 0.0
            for i, batch in enumerate(dataloader):
                input_values = batch[0]
                target_values = batch[1][None, :]
                model.zero_grad()
                outputs = model(input_values, labels=target_values)
                loss = self.label_smoothing_crossentropy(outputs.logits, target_values)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
            avg_loss = total_loss / len(dataloader)
            print(f"Average Loss: {avg_loss}")

if __name__ == "__main__":
    tokenizer = SeamlessM4TTokenizer.from_pretrained("facebook/seamless-m4t-v2")
    model = SeamlessM4TAudioModel.from_pretrained("facebook/seamless-m4t-v2")
    dataloader = DataLoader(batch_size=16, audio_folder="../data", transcripts_file="../data/transcripts.txt")
    seamless_model = SeamlessModel()
    seamless_model.train_loop(model, dataloader, num_epochs=10, learning_rate=5e-4)