
import loralib as lora


from smallwhisper import SmallWhisper, SmallWhisperConfig
from data_loader import WhisperDataLoader


data = WhisperDataLoader().load('/home/chenlong/workspace/services/asr_small/one_batch.hf/', batch_size=2)
device = 'cuda'

model = SmallWhisper(SmallWhisperConfig).from_pretrained("small")
lora.mark_only_lora_as_trainable(model, bias='lora_only')
model.to(device)
print(f"Model loaded and mapped to {device}. Yay")

for batch in data:
    batch = batch.to(device)
    logits, loss = model(**batch)
    print(loss)
    

