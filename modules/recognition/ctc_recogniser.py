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
		
		
		self.encoder =  EncoderFactory.get_encoder(opt)
		
		if self.encoder:
			input_size= opt.hidden_size
		
		if opt.Prediction == 'CTC':
			self.prediction =  nn.Linear(input_size, opt.num_class)
		else:
			raise Exception('incorrect prediction model: {}'.format(opt.Prediction))
	
	def forward(self, x,**kwargs):
		if self.encoder is not None:                   # output of CNN feature extractor. is directly fed to linear # layer.
			x=self.encoder(x,kwargs)
		return self.prediction(x.contiguous())
	
	
