from models.three_stage_model import ThreeStageModel
from models.vit import Vit


class ModelFactory:
	
	@staticmethod
	def getModel(opt, characters):
		if opt.model == "three_stage_model":
			return ThreeStageModel(opt, characters)
		
		if opt.model=="vit":
			return Vit(opt, characters)
		else:
			raise Exception("{} model not defined".format(opt.model))