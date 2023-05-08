import torch
from torch import nn
import torch.nn.functional as F

from ..modules.EncDec import (
    LightConvEncoderLayer,
    LightConvDecoderLayer
)

from ..modules.embedding import EmbeddingBlock


class LightConvModel(nn.Module):
    def __init__(self,
                 path="facebook/bart-large-cnn",
                 path_to_embed="model.encoder.embed_tokens",
                 path_to_learned_pos="model.encoder.embed_positions",
                 n_heads: int = 8,
                 dropout: float = 0.3,
                 encoder_kernels: list = [3, 7, 15, 31, 31, 31, 31],
                 decoder_kernels: list = [3, 7, 15, 31, 31, 31],
                 encoder_dilations: list = None,
                 decoder_dilations: list = None,
                 maxpool: bool = False):

        super().__init__()

        self.embedding = EmbeddingBlock(
            path=path,
            path_to_embed=path_to_embed,
            path_to_learned_pos=path_to_learned_pos
        )
        
        d_model = self.embeddig.embedding_dim
        dim_feedforward = d_model * 2

        # self.dec_embeddig = TripleEmbeddingBlock(
        #     num_word_embeddings=target_vocab_size,
        #     num_type_embeddings=target_max_sentences,
        #     embedding_dim=d_model,
        #     padding_index=padding_index
        # )

        if encoder_dilations:
            self.encoder = nn.Sequential(
                *[LightConvEncoderLayer(d_model=d_model,
                                        n_heads=n_heads,
                                        dim_feedforward=dim_feedforward,
                                        dropout=dropout,
                                        dilation=dilation,
                                        maxpool=maxpool)
                  for dilation in encoder_dilations]
            )
        else:
            self.encoder = nn.Sequential(
                *[LightConvEncoderLayer(d_model=d_model,
                                        n_heads=n_heads,
                                        dim_feedforward=dim_feedforward,
                                        kernel_size=kernel,
                                        dropout=dropout,
                                        maxpool=maxpool)
                  for kernel in encoder_kernels]
            )

        if decoder_dilations:
            self.decoder = nn.ModuleList(
                [LightConvDecoderLayer(d_model=d_model,
                                       n_heads=n_heads,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout,
                                       dilation=dilation)
                 for dilation in decoder_dilations]
            )
        else:
            self.decoder = nn.ModuleList(
                [LightConvDecoderLayer(d_model=d_model,
                                       n_heads=n_heads,
                                       dim_feedforward=dim_feedforward,
                                       kernel_size=kernel,
                                       dropout=dropout)
                 for kernel in decoder_kernels]
            )

        self.classifier = nn.Linear(in_features=d_model,
                                    out_features=self.embeddig.num_embeddings)

    def forward(self,
                source_tokens: torch.tensor,
                target_tokens: torch.tensor):
        enc_embeddings = self.embedding(source_tokens)

        enc_output = self.encoder(enc_embeddings)

        output = self.embedding(target_tokens)

        for decoder in self.decoder:
            output = decoder(output,
                             enc_output)

        return self.classifier(output)
