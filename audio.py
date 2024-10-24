
import torch
from dataclasses import dataclass
from datasets import load_dataset
from transformers import WhisperTokenizer, WhisperFeatureExtractor, WhisperProcessor
import datasets
from datasets import DatasetDict, concatenate_datasets
from multiprocess import set_start_method


@dataclass
class AudioLoaderConfig:
    rand_seed: int = 21_577
    n_batch: int = 512
    n_proc: int = 8
    sample_rate: int = 16000
    max_input_len: int = (30 * 16000)
    min_input_len: int = (0 * 16000)
    max_label_len: int = 448 
    

class AudioLoader:
    def __init__(self, config, model='small', lang='german', task='transcribe'):
        # processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(f"openai/whisper-{model}")
        self.tokenizer = WhisperTokenizer.from_pretrained(f"openai/whisper-{model}", language=lang, task=task)
        self.config = config
    
    def get_all(self):
        raw_ds = self.load_raw()
        dataset = self.process_data(raw_ds)
        return dataset

    def load_raw(self):
        ds1 = load_dataset('facebook/multilingual_librispeech', 'german', cache_dir="/media/hdd/.cache/huggingface", 
                           streaming=False, trust_remote_code=True,)

        ds2 = load_dataset('mozilla-foundation/common_voice_16_0', 'de', cache_dir="/media/hdd/.cache/huggingface", trust_remote_code=True,)

        ds3 = load_dataset('facebook/voxpopuli', 'de', cache_dir="/media/hdd/.cache/huggingface", trust_remote_code=True,)
        
        ds1 = self.rename_columns(ds1, "transcript")
        ds2 = self.rename_columns(ds2, "sentence")
        ds3 = self.rename_columns(ds3, "raw_text")

        dataset = DatasetDict()
        dataset['train'] = concatenate_datasets([ds1['train'], ds2['train'], ds2['validation']])
        dataset['ID_eval'] = concatenate_datasets([ds1['test'], ds2['test']])
        dataset['OOD_eval'] = concatenate_datasets([ds3['validation'], ds3['test']])

        dataset['train'] = dataset['train'].shuffle(seed=self.config.rand_seed)
        dataset['ID_eval'] = dataset['ID_eval'].shuffle(seed=self.config.rand_seed)
        dataset['OOD_eval'] = dataset['OOD_eval'].shuffle(seed=self.config.rand_seed)
        return dataset

    def rename_columns(self, ds, text_col_nm):
        ds = ds.cast_column("audio", datasets.features.Audio(self.config.sample_rate))
        ds = ds.rename_column(text_col_nm, "text")
        dataset_features = ds['train'].features.keys()
        columns_to_keep = {"audio", "text"}
        ds = ds.remove_columns(set(dataset_features - columns_to_keep))
        return ds
   
    def process_data(self, ds):
        def batch_process(batch):
            audio = [sample["array"] for sample in batch["audio"]]
            batch['input_features'] = self.feature_extractor(audio, sampling_rate=16000, ).input_features
            batch["labels"] = self.tokenizer(batch["text"]).input_ids
            # inputs = self.feature_extractor(audio, sampling_rate=16000, device='gpu')
            # batch["input_features"] = inputs.input_features
            batch["input_length"] = [len(sample) for sample in audio]
            # batch["labels"] = self.tokenizer(batch["text"]).input_ids
            return batch
        
        dataset = ds.map(batch_process, batched=True, batch_size=self.config.n_batch, num_proc=self.config.n_proc)

        def is_audio_in_length_range(length):
            return self.config.min_input_len < length < self.config.max_input_len
        
        dataset.filter(function=is_audio_in_length_range, input_columns=["input_length"])
        
        def is_labels_in_length_range(labels):
            return 0 < len(labels) <= self.config.max_label_len
        
        dataset.filter(function=is_labels_in_length_range, input_columns=["labels"])
        # dataset = self.pad_inputs(dataset)
        return dataset
    
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

if __name__ == "__main__":
    set_start_method("spawn")
    dataset = AudioLoader(AudioLoaderConfig).get_all()
    dataset.save_to_disk('/media/hdd/de_dataset', num_proc=8)
