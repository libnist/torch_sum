from transformers import AutoTokenizer
from datasets import load_dataset
import datasets

from torch.utils.data import Dataset, DataLoader

import os

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import torch
from torch.optim.lr_scheduler import LambdaLR


# The class below provides a pretrained tokenizer
class Tokenizer:
    def __init__(self,
                 path,
                 return_tensors="pt",
                 return_attention_mask=False,
                 skip_special_tokens_in_decode=True):
        self.tokenizer = AutoTokenizer.from_pretrained(path)

        self.return_tensors = (return_tensors
                               if return_tensors in ["pt", "np", "tf"]
                               else None)

        self.attention_mask = return_attention_mask

        self.skip_special_tokens_in_decode = skip_special_tokens_in_decode

    def __call__(self, inputs, **kwargs):
        result = self.tokenizer(inputs,
                                return_tensors=self.return_tensors,
                                return_attention_mask=self.attention_mask,
                                **kwargs)
        if len(result.keys()) == 1:
            return result[list(result.keys())[0]]
        return result

    def decode(self, inputs):
        if isinstance(inputs[0], list):
            return self.tokenizer.batch_decode(
                inputs,
                skip_special_tokens=self.skip_special_tokens_in_decode
            )
        else:
            return self.tokenizer.decode(
                inputs,
                skip_special_tokens=self.skip_special_tokens_in_decode
            )


# The function below downloads/caches a dataset from huggingface
def get_data(path: str,
             split: str,
             *args,
             version: str = "3.0.0",
             **kwargs) -> datasets.arrow_dataset.Dataset:
    """Downloads and returns a dataset in a specified version.

    Args:
        path (str): path or name of the dataset.
        split (str): which split to download (e.g. "train", "test", "val")
        version (str, optional): version of the dataset. Defaults to "3.0.0".

    Returns:
        datasets.arrow_dataset.Dataset: A hugging face dataset.
    """
    return load_dataset(path=path,
                        split=split,
                        *args,
                        version=version,
                        **kwargs)

# The class below provides a custom torch.utls.Dataset


class DocumentSummaryDataset(Dataset):
    def __init__(self,
                 documents: list,
                 summaries: list,
                 document_max_tokens: int,
                 summary_max_tokens: int,
                 tokenizer: Tokenizer):

        self._doc_max_tokens = document_max_tokens
        self._sum_max_tokens = summary_max_tokens

        error_message = "[ERROR] Shape missmatch of documents and summaries"
        self._len_documents = len(documents)
        self._len_summaries = len(summaries)
        assert self._len_documents == self._len_documents, error_message

        self._documents = documents
        self._summaries = summaries

        self._tokenizer = tokenizer

    def __len__(self):
        return self._len_documents

    def __getitem__(self, i):
        encoded_document = self._tokenizer(
            self._documents[i],
            padding="max_length",
            max_length=self._doc_max_tokens
        )

        encoded_summaries = self._tokenizer(
            self._summaries[i],
            padding="max_length",
            max_length=self._sum_max_tokens
        )

        return (encoded_document[0][:self._doc_max_tokens],
                encoded_summaries[0][:self._sum_max_tokens-1],
                encoded_summaries[0][1:self._sum_max_tokens])

# The function below provides a custom torch.utils.DataLoader


def get_dataloader(
    documents,
    summaries,
    document_max_tokens: int,
    summary_max_tokens: int,
    tokenizer: Tokenizer,
    batch_size: int,
    num_workers: int = os.cpu_count(),
    shuffle: bool = False
):
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    dataset = DocumentSummaryDataset(
        documents=documents,
        summaries=summaries,
        document_max_tokens=document_max_tokens,
        summary_max_tokens=summary_max_tokens,
        tokenizer=tokenizer
    )
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers)
    return dataloader


def summary_writer(path: str,
                   model_name: str,
                   time: str = None,
                   extra: str = None):
    if not time:
        time = datetime.now().strftime("%Y-%m-%d")

    if not extra:
        log_dir = os.path.join(path, model_name, time)
    else:
        log_dir = os.path.join(path, model_name, time, extra)

    return SummaryWriter(log_dir=log_dir)


def get_transformer_scheduler(optimizer: torch.optim.Optimizer,
                              warmup_steps: int,
                              d_model: int,
                              last_epoch: int = -1):

    warmup_coeff = warmup_steps**-1.5

    # Inverse of the optimizers default lr is used to neutrize the effect of it.
    d_model_coeff = (d_model**-0.5) * (1 / optimizer.param_groups[0]["lr"])

    def lr_lambda(current_step):
        current_step += 1
        return d_model_coeff * min(current_step**-0.5,
                                   current_step * warmup_coeff)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_modified_transformer_scheduler(optimizer: torch.optim.Optimizer,
                                       warmup_steps: int,
                                       coeff: float = 0.002,
                                       last_epoch: int = -1):

    warmup_coeff = warmup_steps**-1.5

    # Inverse of the optimizers default lr is used to neutrize the effect of it.
    coeff = coeff * (1 / optimizer.param_groups[0]["lr"])

    def lr_lambda(current_step):
        current_step += 1
        return coeff * min(current_step**-0.5, current_step * warmup_coeff)

    return LambdaLR(optimizer, lr_lambda, last_epoch)
