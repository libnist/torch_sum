import torch
from torch import nn
from transformers import AutoModelForSeq2SeqLM

from ..utils.input import positional_encoding

class EmbeddingBlock(nn.Module):
    
    def __init__(self, path, path_to_embed, path_to_learned_pos=None):
        super().__init__()
        
        self.module = AutoModelForSeq2SeqLM.from_pretrained(path)
        
        self.embedding = self.attr_from_path(path_to_embed)
        
        self.positional_embedding = (self.attr_from_path(path_to_learned_pos)
        if path_to_learned_pos else None)
        
        self.num_embeddings = self.embedding.num_embeddings
        self.embedding_dim = self.embedding.embedding_dim

        del self.module
        
    def forward(self, x):
        token_length = x.shape[-1]
        
        word_embedding = self.embedding(x)
        
        if self.positional_embedding:
            pos_embed = self.positional_embedding(x)
        else:
            pos_embed = positional_encoding(token_length,
                                            self.embedding_dim,
                                            word_embedding.device,
                                            word_embedding.dtype)
        return word_embedding + pos_embed
        
    def attr_from_path(self, path):
        whole_path = path.split(".")
        module = self.module
        for path in whole_path:
            module = getattr(module, path)
        for param in module.parameters():
          param.requires_grad = False
        return module