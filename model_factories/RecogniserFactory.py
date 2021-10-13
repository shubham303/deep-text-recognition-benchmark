from modules.recognition.attn_recogniser import AttnRecogniser
from modules.recognition.ctc_recogniser import CTCRecogniser
from modules.recognition.transformer_recogniser import TransformerRecogniser, VisionTransformerWithDecoder
from modules.recognition.vision_transformer import VisionTransformerRecogniser

class RecogniserFactory:
	
	@staticmethod
	def get_recogniser(opt, input_size=512):
		if opt.recogniser == "ctc":
			return CTCRecogniser(opt, input_size)
		
		if opt.recogniser == "attn":
			return AttnRecogniser(opt, input_size)
		
		if opt.recogniser == "transformer":
			return TransformerRecogniser(opt, input_size)
		
		if opt.recogniser == "vit":
			return VisionTransformerRecogniser(opt)
		
		if opt.recogniser == "vit_decoder":
			return VisionTransformerWithDecoder(opt)
		else:
			if opt.recogniser == None:
				print("recogniser not declared")
				raise Exception("opt.recogniser value cannot be null")
			else:
				print("{} recogniser not declared".format(opt.recogniser))
				raise Exception("{} recogniser not available".format(opt.recogniser))

