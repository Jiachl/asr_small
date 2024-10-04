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

        self.encoder = nn.ModuleDict(dict(
            conv1 = nn.Conv1d(in_channels=config.n_channel, out_channels=config.n_embd, kernel_size=(3,), stride=(1,), padding=(1,)),
            conv2 = nn.Conv1d(in_channels=config.n_embd, out_channels=config.n_embd, kernel_size=(3,), stride=(2,), padding=(1,)),
            embd_pos = nn.Embedding(config.block_size, config.n_embd),
            encoder = nn.ModuleList([EncBlock(config) for _ in range(config.n_layer)]),
            ln = nn.LayerNorm(config.n_embd),
        ))

        self.decoder = nn.ModuleDict(dict(

        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)


class EncBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc_1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.fc_2 = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.fc_1(x)
        x = self.gelu(x)
        x = self.fc_2(x)
        return x
    

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n