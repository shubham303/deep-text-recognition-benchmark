import math
import torch
from einops.layers.torch import Rearrange
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Transformer, TransformerDecoderLayer, \
	TransformerDecoder, LayerNorm

import configuration

class PatchEmbedding(nn.Module):
	"""
	this is convolution based patch embedding.
	"""
	
	def __init__(self, in_channels: int = 3, patch_size=4, emb_size=512):
		self.patch_size = patch_size
		super().__init__()
		self.projection = nn.Sequential(
			# using a conv layer instead of a linear one -> performance gains
			nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
			Rearrange('b e (h) (w) -> b (h w) e'),
		)
	
	def forward(self, x):
		x = self.projection(x)
		return x
# TODO cls toke is not necessary for text recogntion task. if required pls add here.


class PositionalEncoding(nn.Module):
	def __init__(self,
	             emb_size: int,
	             dropout: float,
	             maxlen: int = 5000):
		super(PositionalEncoding, self).__init__()
		den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
		pos = torch.arange(0, maxlen).reshape(maxlen, 1)
		pos_embedding = torch.zeros((maxlen, emb_size))
		pos_embedding[:, 0::2] = torch.sin(pos * den)
		pos_embedding[:, 1::2] = torch.cos(pos * den)
		pos_embedding = pos_embedding.unsqueeze(-2)
		
		self.dropout = nn.Dropout(dropout)
		self.register_buffer('pos_embedding', pos_embedding)
	
	def forward(self, token_embedding: Tensor):
		return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


def generate_square_subsequent_mask(sz):
	"""

	:param sz: size of mask matrix
	:return: return 2d mask
	"""
	mask = (torch.triu(torch.ones((sz, sz), device=configuration.device)) == 1).transpose(0, 1)
	mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
	return mask


def create_mask(src, tgt):
	# todo make dimension of src is wrong
	src_seq_len = src.shape[2]
	tgt_seq_len = tgt.shape[1]
	tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
	src_mask = torch.zeros((src_seq_len, src_seq_len), device=configuration.device).type(torch.bool)
	# 2: index of padding token in character list.
	tgt_padding_mask = (tgt == 0)
	tgt_padding_mask[:, 0] = False  # dont mask first go symbol
	return src_mask, tgt_mask, tgt_padding_mask


class TokenEmbedding(nn.Module):
	def __init__(self, vocab_size: int, emb_size):
		super(TokenEmbedding, self).__init__()
		self.embedding = nn.Embedding(vocab_size, emb_size)
		self.emb_size = emb_size
	
	def forward(self, tokens: Tensor):
		return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class PositionalEncoding2(nn.Module):
	def __init__(self,
	             emb_size: int,
	             dropout: float,
	             maxlen: int = 5000):
		super(PositionalEncoding2, self).__init__()
		den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
		pos = torch.arange(0, maxlen).reshape(maxlen, 1)
		pos_embedding = torch.zeros((maxlen, emb_size))
		pos_embedding[:, 0::2] = torch.sin(pos * den)
		pos_embedding[:, 1::2] = torch.cos(pos * den)
		pos_embedding = pos_embedding.unsqueeze(-2)
		
		self.dropout = nn.Dropout(dropout)
		self.register_buffer('pos_embedding', pos_embedding)
	
	def forward(self, token_embedding: Tensor):
		return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])
