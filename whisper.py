import torch
import torch.nn as nn
import torch.nn.functional as func


class WhisperConfig:
    block_size: int = 1500
    vocab_size: int = 51865
    n_channel: int = 80
    n_layer: int = 12
    n_head: int = 1
    n_embd: int = 768
    


class Whisper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            conv1 = nn.Conv1d(in_channels=config.n_channel, out_channels=config.n_embd, kernel_size=(3,), stride=(1,), padding=(1,)),
            conv2 = nn.Conv1d(in_channels=config.n_embd, out_channels=config.n_embd, kernel_size=(3,), stride=(2,), padding=(1,)),
            pos_embd = nn.Embedding(config.block_size, config.n_embd),
            head = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln = nn.Linear()
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

