import math

import torch
from einops.layers.torch import Rearrange
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Transformer, TransformerDecoderLayer, \
    TransformerDecoder, LayerNorm

import configuration






class TransformerRecogniser(nn.Module):
	
	def __init__(self,
	             num_encoder_layers: int,
	             num_decoder_layers: int,
	             patch_embedding,
	             positional_encoding,
	             custom_encoder=None,
	             custom_decoder=None,
	             emb_size: int = 512,
	             nhead: int = 8,
	             tgt_vocab_size: int = 25,
	             dim_feedforward: int = 512,
	             dropout: float = 0.1):
		"""

		:param num_encoder_layers:
		:param num_decoder_layers:
		:param custom_encoder:  custom modelled encoder of transformer
		:param custom_decoder:  custom modelled decoder of transformer
		:param emb_size: size of embedding vector
		:param nhead:  number of heads in multihead attention
		:param tgt_vocab_size: number of characters
		:param dim_feedforward: size of output vector
		:param dropout:
		"""
		super(Seq2SeqTransformer, self).__init__()
		self.transformer = Transformer(d_model=emb_size,
		                               nhead=nhead,
		                               num_encoder_layers=num_encoder_layers,
		                               num_decoder_layers=num_decoder_layers,
		                               dim_feedforward=dim_feedforward,
		                               dropout=dropout,
		                               batch_first=True, custom_decoder=custom_decoder,
		                               custom_encoder=custom_encoder)
		self.generator = nn.Linear(emb_size, tgt_vocab_size)
		self.tgt_tok_emb = patch_embedding
		self.positional_encoding = positional_encoding  # this is 1D positional encoding. we can use 2d encoding also
		self.num_classes = tgt_vocab_size
	
	def forward(self,
	            src: Tensor,
	            trg: Tensor,
	            is_train=True,
	            batch_max_len=25,
	            regex=None,
	            character_list=None,
	            tgt_mask: Tensor = None,
	            tgt_padding_mask: Tensor = None,
	            ):
		"""

		:param src:
		:param trg:
		:param is_train:
		:param batch_max_len: length of maximum length word
		:param regex: regex pattern if we need output words to follow certain pattern
		:param character_list:
		:param tgt_mask:
		:param tgt_padding_mask:
		:return:
		"""
		if is_train:
			src_emb = self.positional_encoding(src)
			tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
			
			outs = self.transformer(src_emb, tgt_emb, None, tgt_mask, None,
			                        None, tgt_padding_mask, None)
			return self.generator(outs)
		
		else:
			batch_size = src.size(0)
			num_steps = batch_max_len + 1
			probs = torch.FloatTensor(batch_size, num_steps, self.num_classes).fill_(0).to(configuration.device)
			memory = self.encode(src)
			ys = torch.ones(batch_size, 1).fill_(0).type(torch.long).to(configuration.device)
			
			for i in range(num_steps):
				memory = memory.to(configuration.device)
				tgt_mask = (generate_square_subsequent_mask(ys.size(1))
				            .type(torch.bool)).to(configuration.device)
				
				out = self.decode(ys, memory, tgt_mask)
				
				probs_step = self.generator(out[:, -1])
				
				if regex:  # set probablities of some characters to -infinity as per regex requirement.
					probs_step = self.updateprobs_step(probs_step, i, regex, character_list)
				
				probs[:, i, :] = probs_step
				
				_, next_input = probs_step.max(1)
				# next_input = next_input.item()
				next_input = next_input.resize(batch_size, 1)
				ys = torch.cat([ys,
				                next_input], dim=1)
			
			return probs
	
	def encode(self, src: Tensor):
		return self.transformer.encoder(self.positional_encoding(src))
	
	def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
		return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(
			tgt)), memory,
			tgt_mask)
