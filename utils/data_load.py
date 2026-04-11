import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from datasets import load_dataset

"""
Usage:
--------------------------------------------------------------------------
dm = DataModule(
    repo_id="han2o/grant-ortsaem-processedV2", # hugging face path
    variant="short_gaps",            # short_gap (0.5 to 2.0) or long_gap(0.5 to 5.0)
    input_key="masked_spectrogram",
    target_key="spectrogram",
    mask_key="mask",
    batch_size=16,
    streaming=True,

    # max number of samples to take from dataset
    max_train_samples=10000,         # max number of samples 10000
    max_val_samples=1000,            # max number of sampels 1000
    max_test_samples=1000,           # max number of sampels 1000
    num_workers=2,
    pin_memory=True
)
--------------------------------------------------------------------------
"""

# for non streaming, streaming = False
class Load_SpecDataset(Dataset):
    """
    Build PyTorch Dataset by loading everything from the HuggingFace split into RAM all at once.
    """
    def __init__(self, hf_split, input_key="masked_spectrogram", target_key="spectrogram", mask_key="mask",
                 add_channel_dim=True, dtype=torch.float32):
        self.hf_split = hf_split # split from hugging face load_dataset
        self.input_key = input_key 
        self.target_key = target_key
        self.mask_key = mask_key
        self.add_channel_dim = add_channel_dim
        self.dtype = dtype # tensor type for spectrogram tensors

        # schema check
        first = self.hf_split[0]
        needed = [self.input_key, self.target_key]
        if self.mask_key is not None:
            needed.append(self.mask_key)

        missing = [k for k in needed if k not in first]
        if missing:
            raise KeyError(f"Missing required keys: {missing}")

    def __len__(self):
        return len(self.hf_split)

    def __getitem__(self, idx):
        # get 1 row
        row = self.hf_split[idx]

        # load input and target spectrograms as numpy arrays
        x = np.asarray(row[self.input_key], dtype=np.float32)
        y = np.asarray(row[self.target_key], dtype=np.float32)

        # add channel dimension checks 
        if self.add_channel_dim:
            if x.ndim == 2:
                x = x[None, :, :]
            if y.ndim == 2:
                y = y[None, :, :]

        # create output dictioanry with spectrogram tensors
        item = {
            "x": torch.tensor(x, dtype=self.dtype),
            "y": torch.tensor(y, dtype=self.dtype),
        }

        # load mask
        if self.mask_key is not None:
            m = np.asarray(row[self.mask_key], dtype=np.float32)
            if self.add_channel_dim and m.ndim == 2:
                m = m[None, :, :]
            item["mask"] = torch.tensor(m, dtype=self.dtype)


        # metadata
        for key in [
            "example_id",
            "split",
            "recording_idx",
            "clip_idx",
            "audio_filename",
            "sample_rate",
            "clip_duration_seconds",
            "n_mels",
            "time_frames",
            "hop_length",
            "gap_seconds",
            "gap_frames",
            "mask_start_frame",
            "mask_end_frame",
        ]:
            if key in row:
                item[key] = row[key]

        return item

# for streaming dataset, streaming = True
class Load_IterSpecDataset(IterableDataset):
    """
    Build an iterable PyTorch dataset that converts each row on the fly without loading everything into RAM.
    """
    def __init__(self, hf_split, input_key="masked_spectrogram", target_key="spectrogram", mask_key="mask",
                  add_channel_dim=True, dtype=torch.float32, max_samples=None):
        self.hf_split = hf_split
        self.input_key = input_key
        self.target_key = target_key
        self.mask_key = mask_key
        self.add_channel_dim = add_channel_dim
        self.dtype = dtype
        self.max_samples = max_samples

    # convert one row into tensors
    def convert_row_(self, row):
        x = np.asarray(row[self.input_key], dtype=np.float32)
        y = np.asarray(row[self.target_key], dtype=np.float32)

        if self.add_channel_dim:
            if x.ndim == 2:
                x = x[None, :, :]
            if y.ndim == 2:
                y = y[None, :, :]

        item = {
            "x": torch.tensor(x, dtype=self.dtype),
            "y": torch.tensor(y, dtype=self.dtype),
        }

        if self.mask_key is not None:
            m = np.asarray(row[self.mask_key], dtype=np.float32)
            if self.add_channel_dim and m.ndim == 2:
                m = m[None, :, :]
            item["mask"] = torch.tensor(m, dtype=self.dtype)

        for key in [
            "example_id",
            "split",
            "recording_idx",
            "clip_idx",
            "audio_filename",
            "sample_rate",
            "clip_duration_seconds",
            "n_mels",
            "time_frames",
            "hop_length",
            "gap_seconds",
            "gap_frames",
            "mask_start_frame",
            "mask_end_frame",
        ]:
            if key in row:
                item[key] = row[key]

        return item
    
    # turns huging face streaming split into iterable of tensors for training
    def __iter__(self):
        count = 0
        for row in self.hf_split:
            yield self.convert_row_(row)
            count += 1
            if self.max_samples is not None and count >= self.max_samples:
                break

#  main wrapper class, manages the full process
class DataModule:
    """
    wrapper that loads on dataset variant from hugging face parquet repository:
    variants:
    - short_gaps (0.5 to 2.0)
    - long_gaps (0.5 to 5.0)
    """
    def __init__(
        self,
        repo_id,
        variant,
        input_key="masked_spectrogram",
        target_key="spectrogram",
        mask_key="mask",
        batch_size=8,
        num_workers=0,
        train_pattern="train-*.parquet",
        val_pattern="validation-*.parquet",
        test_pattern="test-*.parquet",
        streaming=True,
        max_train_samples=None,
        max_val_samples=None,
        max_test_samples=None,
        shuffle_train_buffer=256,
        pin_memory=False,
        ):

        self.repo_id = repo_id
        self.variant = str(variant)

        self.input_key = input_key
        self.target_key = target_key
        self.mask_key = mask_key

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_pattern = train_pattern
        self.val_pattern = val_pattern
        self.test_pattern = test_pattern
        self.streaming = streaming

        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.max_test_samples = max_test_samples
        self.shuffle_train_buffer = shuffle_train_buffer
        self.pin_memory = pin_memory

        self.hf_ds = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    # builds hugging face path for repo and gap
    def path_(self, pattern):
        return f"hf://datasets/{self.repo_id}/{self.variant}/{pattern}"

    # load huggingface dataset
    def load_hf(self):
        # data split
        data_files = {
            "train": self.path_(self.train_pattern),
            "validation": self.path_(self.val_pattern),
            "test": self.path_(self.test_pattern),
        }
        # load dataset with streaming option
        self.hf_ds = load_dataset(
            "parquet",
            data_files=data_files,
            streaming=self.streaming,
        )

        return self.hf_ds
    
    # converts huggingface splits into pytorch datasets
    def build_datasets(self):
        if self.hf_ds is None:
            self.load_hf() # load hugging face dataset

        train_split = self.hf_ds["train"]
        val_split = self.hf_ds["validation"]
        test_split = self.hf_ds["test"]

        # streaming on, build iterable datasets that convert each row on the fly
        if self.streaming:
            # shuffle only the streaming train split
            train_split = train_split.shuffle(buffer_size=self.shuffle_train_buffer)

            self.train_dataset = Load_IterSpecDataset(
                train_split,
                input_key=self.input_key,
                target_key=self.target_key,
                mask_key=self.mask_key,
                max_samples=self.max_train_samples
            )

            self.val_dataset = Load_IterSpecDataset(
                val_split,
                input_key=self.input_key,
                target_key=self.target_key,
                mask_key=self.mask_key,
                max_samples=self.max_val_samples
            )

            self.test_dataset = Load_IterSpecDataset(
                test_split,
                input_key=self.input_key,
                target_key=self.target_key,
                mask_key=self.mask_key,
                max_samples=self.max_test_samples
            )

        # streaming off, load everything into RAM and build standard PyTorch datasets
        else:
            if self.max_train_samples is not None:
                train_split = train_split.select(
                    range(min(self.max_train_samples, len(train_split)))
                    )
            if self.max_val_samples is not None:
                val_split = val_split.select(
                    range(min(self.max_val_samples, len(val_split)))
                    )
            if self.max_test_samples is not None:
                test_split = test_split.select(
                    range(min(self.max_test_samples, len(test_split)))
                    )

            self.train_dataset = Load_SpecDataset(
                train_split,
                input_key=self.input_key,
                target_key=self.target_key,
                mask_key=self.mask_key
            )

            self.val_dataset = Load_SpecDataset(
                val_split,
                input_key=self.input_key,
                target_key=self.target_key,
                mask_key=self.mask_key
            )

            self.test_dataset = Load_SpecDataset(
                test_split,
                input_key=self.input_key,
                target_key=self.target_key,
                mask_key=self.mask_key
            )

        return self.train_dataset, self.val_dataset, self.test_dataset

    # build dataloaders from datasets
    def get_dataloaders(self):
        if self.train_dataset is None:
            self.build_datasets() # build datasets from hugging face splits

        # streaming is on, no shuffling at dataloader level since it's handled in the iterable dataset, just batch and load
        if self.streaming:
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )
            test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )

        # streaming is off, standard dataloader with shuffling for train and no shuffling for val/test
        else:
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )
            test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )

        return train_loader, val_loader, test_loader

    # call all steps
    def setup(self):
        return self.get_dataloaders()