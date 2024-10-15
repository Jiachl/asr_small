from datasets import load_dataset
from transformers import WhisperTokenizerFast, WhisperFeatureExtractor
import datasets
from datasets import DatasetDict, concatenate_datasets

@dataclass
class DataLoaderConfig:
    rand_seed: int = 21_577
    n_batch: int = 128
    n_proc: int = 1
    sample_rate: int = 16000
    max_input_len: int = (30 * 16000)
    min_input_len: int = (0 * 16000)
    max_label_len: int = 448 
    

class DataLoader:
    def __init__(self, config):
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained("distil-whisper/distil-medium.en")
        self.tokenizer = WhisperTokenizerFast.from_pretrained("distil-whisper/distil-medium.en")
        self.config = config
    
    def get_all(self):
        raw_ds = self.load_raw()
        dataset = self.process_data(raw_ds)
        return dataset

    def load_raw(self):
        ds1 = load_dataset('facebook/multilingual_librispeech', 'german',
                           cache_dir="/media/hdd/.cache/huggingface",
                           # token=token,
                           streaming=False, trust_remote_code=True,
                           )

        ds2 = load_dataset('mozilla-foundation/common_voice_16_0', 'de',
                            cache_dir="/media/hdd_old/.cache/huggingface"
                            )

        ds3 = load_dataset('facebook/voxpopuli', 'de',
                           cache_dir="/media/hdd_old/.cache/huggingface",
                           )
        
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
            inputs = self.feature_extractor(audio, sampling_rate=16000, device='cuda')
            batch["input_features"] = inputs.input_features
            batch["input_length"] = [len(sample) for sample in audio]
            batch["labels"] = self.tokenizer(batch["text"]).input_ids
            return batch
        
        dataset = dataset.map(ds, batched=True, batch_size=self.config.n_batch, num_proc=self.config.n_proc)
        
        def is_audio_in_length_range(length):
            return self.config.min_input_len < length < self.config.max_input_len
        
        dataset.filter(function=is_audio_in_length_range, input_columns=["input_length"])
        
        def is_labels_in_length_range(labels):
            return 0 < len(labels) <= self.config.max_label_len
        
        dataset.filter(function=is_labels_in_length_range, input_columns=["labels"])
        
        return dataset

