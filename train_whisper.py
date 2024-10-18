import torch
import loralib as lora

from smallwhisper import SmallWhisper, SmallWhisperConfig
from data_loader import WhisperDataLoader
from data_loader_test import DataLoaderTest
from datasets import load_from_disk

import time


device = 'cuda'
torch.cuda.empty_cache()
# data = WhisperDataLoader().load('/home/user/workspace/services/asr_small/one_batch.hf/', batch_size=64)
data = load_from_disk('/home/chenlong/workspace/services/asr_small/one_batch.hf')
dataset = DataLoaderTest("small").get_dataset(sample_ds=data.take(8))
batch = dataset.to(device)

model = SmallWhisper(SmallWhisperConfig).from_pretrained("small", override_args={'use_lora': True})

# model.to(device)
# model.eval()
# with torch.no_grad():
#     logits, loss = model(**batch)
# print(f"loss: {loss.item()}")

lora.mark_only_lora_as_trainable(model, bias='lora_only')
model.to(device)
print(f"Model loaded and mapped to {device}. Yay")

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)
model = torch.compile(model)
torch.set_float32_matmul_precision('high')

for i in range(100):
    t0 = time.time()
    optimizer.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(**batch)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    tokens_per_sec = batch['decoder_input_ids'].shape[0] * batch['decoder_input_ids'].shape[1] / (t1 - t0)
    print(f"step {i}, loss: {loss.item()}, time: {(t1 - t0)*1000:.2f}ms, tok/sec: {tokens_per_sec:.2f}")
