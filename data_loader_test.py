
from transformers import WhisperFeatureExtractor, WhisperTokenizer
from datasets import load_from_disk, Audio
import torch
from dataclasses import dataclass


class DataLoaderTest:
    def __init__(self, model, lang='german', task='transcribe'):
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(f"openai/whisper-{model}")
        self.tokenizer = WhisperTokenizer.from_pretrained(f"openai/whisper-{model}", language=lang, task=task)
        self.decoder_start_token_id = self.tokenizer.get_vocab()["<|startoftranscript|>"]
        self.max_target_length = 448

    def get_dataset(self, sample_ds=None, num_proc=1):
        def prepare_dataset(batch):
            audio = batch['audio']
            batch['input_features'] = self.feature_extractor(audio['array'], sampling_rate=audio["sampling_rate"]).input_features[0]
            batch["labels"] = self.tokenizer(batch["text"]).input_ids
            return batch
        
        if sample_ds is None:
            sample_ds = load_from_disk('/Users/xinzheng/workspace/chenlong/services/asr_small/one_batch.hf')

        sample_ds = sample_ds.cast_column("audio", Audio(sampling_rate=16000))
        sample_ds = sample_ds.map(prepare_dataset, remove_columns=['audio', 'text'], num_proc=num_proc)

        sample_ds = self.pad_inputs(sample_ds)
        return sample_ds
    
    def pad_inputs(self, dataset):
        input_features = [{"input_features": ds["input_features"]} for ds in dataset]
        batch = self.feature_extractor.pad(input_features, return_tensors="pt")
        

        label_features = [{"input_ids": ds["labels"]} for ds in dataset]
        labels_batch = self.tokenizer.pad(label_features, max_length=self.max_target_length, padding='max_length', return_tensors="pt")

        labels = labels_batch["input_ids"]
        decoder_input_ids = labels[:, :-1]
        labels = labels[:, 1:]
        labels_mask = labels_batch.attention_mask[:, 1:]

        # replace padding with -100 to ignore correctly when computing the loss
        labels = labels.masked_fill(labels_mask.ne(1), -100)

        # replace initial prompt tokens with -100 to ignore correctly when computing the loss
        bos_index = torch.argmax((labels == self.decoder_start_token_id).long(), dim=1)
        bos_index = torch.where(bos_index > 0, bos_index + 1, bos_index)
        prompt_mask = torch.arange(labels.shape[1]) < bos_index[:, None]
        labels = torch.where(prompt_mask, -100, labels)

        batch["labels"] = labels.type(torch.long)
        batch['decoder_input_ids'] = decoder_input_ids.type(torch.long)
        return batch