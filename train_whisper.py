import torch
import loralib as lora

from smallwhisper import SmallWhisper, SmallWhisperConfig
from data_loader import WhisperDataLoader
from data_loader_test import DataLoaderTest
from datasets import load_from_disk

import time


device = 'cuda'
torch.cuda.empty_cache()
total_batch_size = 256
B = 8
# data = WhisperDataLoader().load('/home/user/workspace/services/asr_small/one_batch.hf/', batch_size=64)
data = load_from_disk('/home/chenlong/workspace/services/asr_small/one_batch.hf')
dataset = DataLoaderTest("small").get_dataset(sample_ds=data.take(B))
batch = dataset.to(device)
grad_accum_steps = total_batch_size // B

model = SmallWhisper(SmallWhisperConfig).from_pretrained("small", override_args={'use_lora': True})

# model.to(device)
# model.eval()
# with torch.no_grad():
#     logits, loss = model(**batch)
# print(f"loss: {loss.item()}")

lora.mark_only_lora_as_trainable(model, bias='lora_only')
model.to(device)
print(f"Model loaded and mapped to {device}. Yay")
    
# optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4, betas=(0.9, 0.98), eps=1e-6)
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=3e-4, device_type=device)
model = torch.compile(model)
torch.set_float32_matmul_precision('high')

for i in range(100):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(**batch)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    tokens_per_sec = batch['decoder_input_ids'].shape[0] * batch['decoder_input_ids'].shape[1] / (t1 - t0)
    print(f"step {i} | loss: {loss.item():.6f} | norm: {norm:.4f} | time: {(t1 - t0)*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
