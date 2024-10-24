import torch
import loralib as lora

from smallwhisper import SmallWhisper, SmallWhisperConfig
from data_loader import WhisperDataLoader
from datasets import load_from_disk

import time
import os


device = 'cuda'
torch.cuda.empty_cache()

total_batch_size = 256
B = 64
grad_accum_steps = total_batch_size // B

data_fp = '/media/hdd/de_dataset'
data = load_from_disk(data_fp)
train_ds = WhisperDataLoader().load(data['train'], batch_size=B)
eval_ds = WhisperDataLoader().load(data['ID_eval'], batch_size=B)
# test_ds = WhisperDataLoader().get_dataset(data['OOD_eval'], batch_size=B)

torch.set_float32_matmul_precision('high')
model = SmallWhisper(SmallWhisperConfig).from_pretrained("small", override_args={'use_lora': True})
lora.mark_only_lora_as_trainable(model, bias='lora_only')
model.to(device)
model = torch.compile(model)
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=5e-4, device_type=device)

max_lr = 5e-4
min_lr = 0
warmup_steps = 50
max_steps = 9000

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass


def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    
    if it > max_steps:
        return min_lr
    
    decay_ratio = (max_lr - min_lr) / (max_steps - warmup_steps)
    return max_lr - decay_ratio * (it - warmup_steps)


@torch.no_grad()
def eval_loss():
    model.eval()
    eval_iter = iter(eval_ds)
    val_loss_accum = 0.0
    val_steps = 20
    for i in range(val_steps):
        batch = next(eval_iter)
        batch = batch.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            _, loss = model(**batch)
        loss = loss / val_steps
        val_loss_accum += loss.detach()

    print(f"validation loss: {val_loss_accum.item():.4f}")
    with open(log_file, "a") as f:
        f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            
    model.train()
    return val_loss_accum


def save_checkpoint(step, val_loss_accum):
    # optionally write model checkpoints
    checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
    checkpoint = {
        'model': model.state_dict(),
        'config': model.config,
        'step': step,
        'val_loss': val_loss_accum.item()
    }
    # you might also want to add optimizer.state_dict() and
    # rng seeds etc., if you wanted to more exactly resume training
    torch.save(checkpoint, checkpoint_path)


train_iter = iter(train_ds)
it = 0
reset_it = len(train_iter)

for step in range(max_steps):
    t0 = time.time()
    if step % 250 == 0 or step == max_steps - 1:
        val_loss = eval_loss()
    
    if (step > 0 and step % 1000 == 0) or step == max_steps - 1:
        save_checkpoint(step, val_loss)

    loss_accum = 0.0
    optimizer.zero_grad()
    for micro_step in range(grad_accum_steps):
        batch = next(train_iter)
        batch = batch.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(**batch)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
        it += 1
        if it == reset_it:
            train_iter = iter(train_ds)
            it = 0
    norm = torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()

    if device == 'cuda':
        torch.cuda.synchronize()
    t1 = time.time()

    tokens_per_sec = batch['decoder_input_ids'].shape[0] * batch['decoder_input_ids'].shape[1] * grad_accum_steps/ (t1 - t0)
    print(f"step {step} | loss: {loss.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | time: {(t1 - t0)*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
    with open(log_file, "a") as f:
        f.write(f"{step} train {loss_accum.item():.6f}\n")

    