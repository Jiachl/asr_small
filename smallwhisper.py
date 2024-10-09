from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as func


@dataclass
class SmallWhisperConfig:
    enc_block_size: int = 1500
    dec_block_size: int = 448
    vocab_size: int = 51865
    n_channel: int = 80
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    


class SmallWhisper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = nn.ModuleDict(dict(
            conv1 = nn.Conv1d(in_channels=config.n_channel, out_channels=config.n_embd, kernel_size=(3,), stride=(1,), padding=(1,)),   # (B, C, T)
            conv2 = nn.Conv1d(in_channels=config.n_embd, out_channels=config.n_embd, kernel_size=(3,), stride=(2,), padding=(1,)),  # (B, C, T)
            positional_embedding = nn.Embedding(config.enc_block_size, config.n_embd),
            blocks = nn.ModuleList([EncBlock(config) for _ in range(config.n_layer)]),
            ln_post = nn.LayerNorm(config.n_embd),
            ))
        
        self.decoder = nn.ModuleDict(dict(
            token_embedding = nn.Embedding(config.vocab_size, config.n_embd),
            positional_embedding = nn.Embedding(config.dec_block_size, config.n_embd),
            blocks = nn.ModuleList([DecBlock(config) for _ in range(config.n_layer)]),
            ln = nn.LayerNorm(config.n_embd),
        ))
        # self.proj_out = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, enc_idx, dec_idx):
        B, C, T = enc_idx.size()
        assert T // 2 <= self.config.enc_block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.enc_block_size}"
        enc_tok_emb = func.gelu(self.encoder.conv1(enc_idx))
        enc_tok_emb = func.gelu(self.encoder.conv2(enc_tok_emb))
        enc_tok_emb = enc_tok_emb.transpose(1, 2)
        enc_pos = torch.arange(0, enc_tok_emb.shape[1], dtype=torch.long, device=enc_idx.device)
        enc_pos_emb = self.encoder.positional_embedding(enc_pos) # position embeddings of shape (T, n_embd)
        x = enc_tok_emb + enc_pos_emb
        for block in self.encoder.blocks:
            x = block(x)
        x = self.encoder.ln_post(x)

        B, D = dec_idx.size()
        assert D <= self.config.dec_block_size, f"Cannot forward sequence of length {D}, block size is only {self.config.dec_block_size}"
        dec_pos = torch.arange(0, D, dtype=torch.long, device=dec_idx.device)
        dec_tok_emb = self.decoder.token_embedding(dec_idx) # token embeddings of shape (B, D, n_embd)
        dec_pos_emb = self.decoder.positional_embedding(dec_pos) # position embeddings of shape (D, n_embd)
        y = dec_tok_emb + dec_pos_emb
        for block in self.decoder.blocks:
            y = block(y, x)
        y = self.decoder.ln(y)
        logits = y @ torch.transpose(self.decoder.token_embedding.weight, 0, 1)
        return logits
    

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'small'}
        override_args = override_args or {}
        assert all(k == 'dropout' for k in override_args)
        from transformers import WhisperForConditionalGeneration
        import whisper

        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'small': dict(n_layer=12, n_head=12, n_embd=768),  # 242 params
        }[model_type]
        config_args['vocab_size'] = 51865 
        config_args['enc_block_size'] = 1500
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        config = SmallWhisperConfig(**config_args)
        model = SmallWhisper(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in model.state_dict() if not k.endswith('attn.tril')] # discard this mask / buffer, not a param

        # model_hf = WhisperForConditionalGeneration.from_pretrained(model_type)
        # sd_ref = model_hf.state_dict()
        model_openai = whisper.load_model("small")
        sd_ref = model_openai.state_dict()

        sd_keys_ref = sd_ref.keys()
        assert len(sd_keys_ref) == len(sd_keys), "number of layers not match"
        for name in sd_keys:
            if name.endswith('positional_embedding.weight'):
                ref_name = name[:-27] + 'positional_embedding'
            elif name.endswith('mlp.fc1.weight'):
                ref_name = name[:-14] + 'mlp.0.weight'
            elif name.endswith('mlp.fc2.weight'):
                ref_name = name[:-14] + 'mlp.2.weight'
            elif name.endswith('mlp.fc1.bias'):
                ref_name = name[:-12] + 'mlp.0.bias'
            elif name.endswith('mlp.fc2.bias'):
                ref_name = name[:-12] + 'mlp.2.bias'
            else:
                ref_name = name
            
            assert sd_ref[ref_name].shape == sd[name].shape, f"{ref_name} and {name} size not match"
            with torch.no_grad():
                sd[name].copy_(sd_ref[ref_name])
        return model
    

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1d(x).transpose(1, 2)
        return x
    
class EncBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attn_ln = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.mlp_ln = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        
    def forward(self, x):
        x = x + self.attn(self.attn_ln(x))
        x = x + self.mlp(self.mlp_ln(x))
        return x


class DecBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attn_ln = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config, is_causal=True, is_enc=False)
        self.cross_attn_ln = nn.LayerNorm(config.n_embd)
        self.cross_attn = CrossAttention(config)
        self.mlp_ln = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, x_enc):
        x = x + self.attn(self.attn_ln(x))
        x = x + self.cross_attn(self.cross_attn_ln(x), x_enc)
        x = x + self.mlp(self.mlp_ln(x))
        return x


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x
    
class SelfAttention(nn.Module):
    def __init__(self, config, is_causal=False, is_enc=True):
        super().__init__()
        self.n_head = config.n_head
        self.embd = config.n_embd
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.out = nn.Linear(config.n_embd, config.n_embd)
        self.is_causal = is_causal
        if is_causal:
            block_size = config.enc_block_size if is_enc else config.dec_block_size
            self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
    
    def forward(self, x):
        B, T, C = x.size()
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        scale = (k.shape[-1]) ** (-0.5)
        attn = q @ k.transpose(-2, -1) * scale  # (B, nh, T, T)
        if self.is_causal:
            attn = attn.masked_fill(self.tril[:, :, :T, :T] == 0, float('-inf'))
        attn = func.softmax(attn, dim=-1)
        y = attn @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out(y)  # (B, T, C)
        return y


class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.embd = config.n_embd
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.out = nn.Linear(config.n_embd, config.n_embd)
            
    def forward(self, x, enc_out):
        B, D, C = x.size()
        q = self.query(x).view(B, D, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, D, hs)
        k = self.key(enc_out).view(B, enc_out.size(1), self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(enc_out).view(B, enc_out.size(1), self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        scale = (k.shape[-1]) ** (-0.5)
        attn = q @ k.transpose(-2, -1) * scale  # (B, nh, D, T)
        attn = func.softmax(attn, dim=-1)
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, D, C)
        y = self.out(y)
        return y
    

    