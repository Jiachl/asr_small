import torch
import loralib as lora

from smallwhisper import SmallWhisper, SmallWhisperConfig
from data_loader import WhisperDataLoader
from data_loader_test import DataLoaderTest
from datasets import load_from_disk


# data = WhisperDataLoader().load('/home/user/workspace/services/asr_small/one_batch.hf/', batch_size=64)
data = load_from_disk('/home/user/workspace/services/asr_small/one_batch.hf')
dataset = DataLoaderTest("small").get_dataset(sample_ds=data.take(16))
device = 'cuda'

model = SmallWhisper(SmallWhisperConfig).from_pretrained("small", override_args={'use_lora': True})
lora.mark_only_lora_as_trainable(model, bias='lora_only')
model.to(device)
print(f"Model loaded and mapped to {device}. Yay")

batch = dataset.to(device)

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)

for i in range(100):
    optimizer.zero_grad()
    logits, loss = model(**batch)
    print(f"step {i}, loss: {loss.item()}")
    loss.backward()
    optimizer.step()


    

