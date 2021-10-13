from torch import nn

from model_factories.ModelFactory import EncoderFactory
from modules.sequence_modeling import BidirectionalLSTM

"""
CTC based text recognition model.
Available options for encoder are LSTM.
"""
class CTCRecogniser(nn.Module):
	
	def __init__(self,opt, input_size):
		super().__init__()
		
		self.encoder =  EncoderFactory.get_encoder(opt,input_size)
		
		if self.encoder:
			input_size= opt.hidden_size
		
		
		self.prediction =  nn.Linear(input_size, opt.num_class)

	
	def forward(self,x):
		if self.encoder is not None:                   # output of CNN feature extractor. is directly fed to linear # layer.
			x=self.encoder(x)
		return self.prediction(x.contiguous())
	
	
