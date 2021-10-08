import math

import torch
from einops.layers.torch import Rearrange
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Transformer, TransformerDecoderLayer, \
    TransformerDecoder, LayerNorm

import configuration


class VisionTransformerRecogniser(nn.Module):
	def __init__(self, opt,input_size, dropout=0.1 ):
		"""
		emb_size : emb_size is dimension of input vector to Vit.
		"""
		patch_embedding = PatchEmbedding(opt.input_channel, opt.patch_size, opt.output_channel)
		positional_encoding = PositionalEncoding(
			opt.emb_size, dropout=dropout)
		self.vit = VisionTransformer(opt.encoder_count,patch_embedding, positional_encoding,emb_size=input_size,
		                             nhead=opt.attention_heads,tgt_vocab_size=opt.num_class,
		                             dim_feedforward=opt.hidden_size)
	
	def forward(self,x):
		return self.vit(x)
	
	

class VisionTransformer(nn.Module):
	def __init__(self, num_encoder_layers, patch_embedding, positional_encoding, custom_encoder=None, emb_size=512,
	             nhead=8, tgt_vocab_size: int = 25, dim_feedforward: int = 512, dropout: float = 0.1):
		
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
		
		out = self.encoder(self.positional_encoding(self.patch_embedding(src)), mask=src_mask,
		                   src_key_padding_mask=src_key_padding_mask)
		out = self.generator(out)
		return out

class PatchEmbedding(nn.Module):
	"""
	this is convolution based patch embedding.
	"""
	def __init__(self, in_channels:int=3, patch_size=16, emb_size=512):
	    self.patch_size= patch_size
	    super().__init__()
	    self.projection=nn.Sequential(
	        nn.Conv2d(in_channels,emb_size,kernel_size=patch_size, stride=patch_size ),       # convolution on each image path with kernel size of stride = patch_size. this gives 2d feature embedding for each patch
	        Rearrange('b e (h) (w) -> b (h w) e')  # convert 2d feature map to vector
	    )
	
	def forward(self, x):
	    x = self.projection(x)
	    return x
	    #TODO cls toke is not necessary for text recogntion task. if required pls add here.
	

class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
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
