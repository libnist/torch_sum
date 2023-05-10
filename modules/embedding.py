import torch
from torch import nn
from transformers import AutoModelForSeq2SeqLM

from ..utils.input import positional_encoding


class EmbeddingBlock(nn.Module):

    def __init__(self, path, path_to_embed, path_to_learned_pos="const"):
        super().__init__()

        self.module = AutoModelForSeq2SeqLM.from_pretrained(path)

        self.embedding = self.attr_from_path(path_to_embed)

        if path_to_learned_pos in ("const", "whole"):
            self.positional_embedding = path_to_learned_pos
        else:
            self.positional_embedding = self.attr_from_path(
                path_to_learned_pos)

        self.num_embeddings = self.embedding.num_embeddings
        self.embedding_dim = self.embedding.embedding_dim

        del self.module

    def forward(self, x):
        token_length = x.shape[-1]

        word_embedding = self.embedding(x)

        if isinstance(self.positional_embedding, str):
            if self.positional_embedding == "const":
                word_embedding += positional_encoding(token_length,
                                                      self.embedding_dim,
                                                      word_embedding.device,
                                                      word_embedding.dtype)
        else:
            word_embedding += self.positional_embedding(x)
        return word_embedding

    def attr_from_path(self, path):
        whole_path = path.split(".")
        module = self.module
        for path in whole_path:
            module = getattr(module, path)
        for param in module.parameters():
            param.requires_grad = False
        return module
