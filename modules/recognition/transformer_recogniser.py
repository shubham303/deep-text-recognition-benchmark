import math

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Transformer, TransformerDecoderLayer, \
	TransformerDecoder, LayerNorm

import configuration


class TransformerRecogniser(nn.Module):
	def __init__(self, opt, input_size, dropout=0.1):
		"""
		opt : contains arguments passed to the program
		input_size= size of input feature vectors, input_size depends on size of output of last layer.
		"""
		super().__init__()
		patch_emb = TokenEmbedding(opt.num_class, input_size)
		positional_encoding = PositionalEncoding(
			opt.emb_size, dropout=dropout)
		self.Prediction = Seq2SeqTransformer(opt.encoder_count,
		                                     opt.decoder_count,
		                                     patch_emb,
		                                     positional_encoding,
		                                     input_size,
		                                     opt.attention_heads,
		                                     opt.num_class,
		                                     opt.hidden_size
		                                     )
		for p in self.Prediction.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p)
	
	def forward(self, x):
		return self.Prediction(x.contiguous())


class Seq2SeqTransformer(nn.Module):
	
	def __init__(self,
	             num_encoder_layers: int,
	             num_decoder_layers: int,
	             patch_embedding,
	             positional_encoding,
	             emb_size: int = 512,
	             nhead: int = 8,
	             tgt_vocab_size: int = 25,
	             dim_feedforward: int = 512,
	             dropout: float = 0.1,
	             custom_encoder = None,
		        custom_decoder = None,
	
	):
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


class TokenEmbedding(nn.Module):
	def __init__(self, vocab_size: int, emb_size):
		super(TokenEmbedding, self).__init__()
		self.embedding = nn.Embedding(vocab_size, emb_size)
		self.emb_size = emb_size
	
	def forward(self, tokens: Tensor):
		return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


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