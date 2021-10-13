from torch import nn

from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor

from modules.sequence_modeling import BidirectionalLSTM
from modules.transformation import TPS_SpatialTransformerNetwork

class TransformationModelFactory():
	def __init__(self):
		super(TransformationModelFactory, self).__init__()
	
	@staticmethod
	def get_transformation_model(opt):
		if opt.Transformation ==None:
			print("no transformation module specified")
			return None
		
		if opt.Transformation == "TPS":
			return TPS_SpatialTransformerNetwork(
				F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW),
				I_channel_num=opt.input_channel)
		
		else:
			print("{} transformation module not defined",format(opt.Transformation))
			return None

class FeatureExtractorFactory():
	def __init__(self):
		super(FeatureExtractorFactory, self).__init__()
		
	
	@staticmethod
	def get_feature_extractor_model(opt):
	
		if opt.FeatureExtraction == 'VGG':
			return VGG_FeatureExtractor(opt.input_channel, opt.output_channel)
		elif opt.FeatureExtraction == 'RCNN':
			return RCNN_FeatureExtractor(opt.input_channel, opt.output_channel)
		elif opt.FeatureExtraction == 'ResNet':
			return ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
			
		else:
			if opt.FeatureExtraction ==None:
				#TODO throw error for recognition models which require feature extraction model , e.g lstm ,
				# ctc based seqtoseq model need output from cnn model first.
				print("no feature extraction model is specified")
			else:
				print("{} feature extraction model is not available".format(opt.FeatureExtraction))
			
			return None
		
class EncoderFactory:
	@staticmethod
	def get_encoder(opt, input_size):
		if opt.SequenceModeling == "bilstm":
			return nn.Sequential(
                BidirectionalLSTM(input_size, opt.hidden_size, opt.hidden_size),
                BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
		return None
	
