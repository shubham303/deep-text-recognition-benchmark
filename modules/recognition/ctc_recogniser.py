from torch import nn, Tensor

"""
CTC based text recognition model.
Available options for encoder are LSTM.
"""
class CTCRecogniser(nn.Module):
	
	def __init__(self, encoder: Tensor):
		super().__init__()
		self.encoder= encoder
	
	
	def forward(self, x):
		return self.encoder(x)
	
