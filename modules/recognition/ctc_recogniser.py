from torch import nn, Tensor

"""
CTC based text recognition model.
Available options for encoder are LSTM.
"""
class CTCRecogniser(nn.Module):
	
	def __init__(self,opt,encoder):
		super().__init__()
		self.encoder= encoder
		self.prediction =  nn.Linear(self.SequenceModeling_output, opt.num_class)
	
	def forward(self, x):
		if self.encoder is not None:                   # output of CNN feature extractor. is directly fed to linear # layer.
			x=self.encoder(x)
		return self.prediction(x)
	
	
