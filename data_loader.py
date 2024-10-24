
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import numpy as np

import torch
import torch.nn.functional as func
from torch.utils.data import DataLoader
from transformers import WhisperProcessor, WhisperTokenizer, WhisperFeatureExtractor
from datasets import load_from_disk, Audio



class WhisperDataLoader:
    def __init__(self):
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(f"openai/whisper-small")
        self.tokenizer = WhisperTokenizer.from_pretrained(f"openai/whisper-small", language='german', task='transcribe')
        self.decoder_start_token_id = self.processor.tokenizer.get_vocab()["<|startoftranscript|>"]
        self.decoder_prev_token_id = self.processor.tokenizer.get_vocab()["<|startofprev|>"]

    def load(self, dataset, batch_size=128):        
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=self.processor,
            decoder_start_token_id=self.decoder_start_token_id,
            decoder_prev_token_id=self.decoder_prev_token_id,
            input_padding="longest",
            target_padding="max_length",
            max_target_length=448,
        )
        
        train_dataloader = DataLoader(dataset, collate_fn=data_collator, batch_size=batch_size,)
        return train_dataloader


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int
    decoder_prev_token_id: int
    input_padding: Union[bool, str] = "max_length"
    target_padding: Union[bool, str] = "max_length"
    max_target_length: Optional[int] = None
    feature_extractor = WhisperFeatureExtractor.from_pretrained(f"openai/whisper-small")
    tokenizer = WhisperTokenizer.from_pretrained(f"openai/whisper-small", language='german', task='transcribe')

    def __call__(self, features: List[Dict[str, Union[List[int], np.ndarray]]]) -> Dict[str, np.ndarray]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods

        # dataloader returns a list of features which we convert to a dict
        input_features = {"input_features": [feature["input_features"] for feature in features]}
        label_features = {"input_ids": [feature["labels"] for feature in features]}

        # reformat list to dict and set to pytorch format
        batch = self.feature_extractor.pad(
            input_features,
            padding=self.input_padding,
            return_tensors="pt",
        )

        labels_batch = self.tokenizer.pad(
            label_features,
            max_length=self.max_target_length,
            padding=self.target_padding,
            return_tensors="pt",
        )

        # shift labels to the right to get decoder input ids
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

        batch["labels"] = labels
        batch["decoder_input_ids"] = decoder_input_ids

        return batch
    
    
if __name__ == "__main__":
    train_data = WhisperDataLoader().load('/home/chenlong/workspace/services/asr_small/one_batch.hf/')
    print("Data loaded")