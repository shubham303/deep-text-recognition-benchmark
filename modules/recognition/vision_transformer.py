import math

import torch
from einops.layers.torch import Rearrange
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Transformer, TransformerDecoderLayer, \
	TransformerDecoder, LayerNorm

import configuration
from modules.recognition.utils import PatchEmbedding, PositionalEncoding


class VisionTransformerRecogniser(nn.Module):
	def __init__(self, opt, dropout=0.1):
		
		super().__init__()
		patch_embedding = PatchEmbedding(opt.input_channel, opt.patch_size, opt.emb_size)
		positional_encoding = PositionalEncoding(
			opt.output_channel, dropout=dropout)
		self.vit = VisionTransformer(opt.encoder_count, patch_embedding, positional_encoding, emb_size=opt.emb_size,
		                             nhead=opt.attention_heads, tgt_vocab_size=opt.num_class,
		                             dim_feedforward=opt.emb_size)
	
	def forward(self, x,seqlen):
		return self.vit(x,seqlen)


class VisionTransformer(nn.Module):
	def __init__(self, num_encoder_layers, patch_embedding, positional_encoding, custom_encoder=None,
	             custom_decoder=None, emb_size=512, nhead=8, tgt_vocab_size: int = 25, dim_feedforward: int = 512,
	             dropout: float = 0.1):
		
		
		# I have passed few required parameters, for all set of parameteres refere torch.nn.transformer documentation.
		super().__init__()
		encoder_layer = TransformerEncoderLayer(emb_size, nhead, dim_feedforward, dropout, batch_first=True)
		encoder_norm = LayerNorm(emb_size)
		if custom_encoder is None:
			self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
		else:
			# NOTE: custom encoder should have same parameters as of TransformerEncoder
			self.encoder = custom_encoder
	
		self.generator = nn.Linear(emb_size, tgt_vocab_size)
		self.patch_embedding = patch_embedding
		self.positional_encoding = positional_encoding
		self.num_classes = tgt_vocab_size
	
	def forward(self, src: Tensor,
	            seqlen=25,
	            regex=None,
	            character_list=None,
	            src_mask=None,
	            src_key_padding_mask=None):
		"""

		:param src:
		:param regex:regex pattern if we need output words to follow certain pattern
		:param character_list: list of characters in language
		:param src_mask: attention mask for src variables
		:param src_key_padding_mask : attention mask for src padding
		:return:
		"""
		
		x = self.encoder(self.positional_encoding(self.patch_embedding(src)), mask=src_mask,
		                   src_key_padding_mask=src_key_padding_mask)
		x = x[:, :seqlen]
		
		# batch, seqlen, embsize
		out = self.generator(x)
		return out


