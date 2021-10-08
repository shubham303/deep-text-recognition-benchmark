"""
Copyright (c) 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch.nn as nn

from modules.recognition.transformer_recogniser import Seq2SeqTransformer, create_mask
from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from modules.recognition.attn_recogniser import Attention
from modules.sequence_modeling import BidirectionalLSTM
from modules.transformation import TPS_SpatialTransformerNetwork
from model_factories.ModelFactory import TransformationModelFactory, FeatureExtractorFactory, EncoderFactory, \
	RecogniserFactory


class FourStageModel(nn.Module):
	
	def __init__(self, opt, character):
		super(FourStageModel, self).__init__()
		self.opt = opt
		# self.character is passed to prediction model to mask certain characters during prediction of regex based text
		self.character = character
		self.stages = {'Trans': opt.Transformation, 'Feat': opt.FeatureExtraction,
		               'Seq': opt.SequenceModeling, 'Pred': opt.Prediction}
		
		
		""" Transformation """
		self.Transformation = TransformationModelFactory.get_transformation_model(opt)
		
		""" FeatureExtraction """
		self.FeatureExtraction = FeatureExtractorFactory.get_feature_extractor_model(opt)
		if self.FeatureExtraction is None:
			print("feature extraction model cannot be null for this model")
			raise Exception("Feature extractor is None or not defined.  Feature Extractor={}".format(
				opt.FeatureExtraction))
		self.FeatureExtraction_output = opt.output_channel  # int(imgH/16-1) * 512
		self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1
		
		"""Recogniser"""
		self.recogniser = RecogniserFactory.get_recogniser(opt, self.FeatureExtraction_output)

	
	# regex is used if we expect predicted text to follow certain pattern. ex: regex for PAN number is "[A-Z]{5}[
	# 0-9]{4}[A-Z]{1}" so here for first five positions, predicted probablities of numbers and special characters
	# are set to -infinity.
	def forward(self, input, text, is_train=True, regex=None):
		
		""" Transformation stage """
		if not self.stages['Trans'] == "None":
			input = self.Transformation(input)
			
			
		if self.Transformation is not None:
			input=self.Transformation(input)
			
		
		
		""" Feature extraction stage """
		visual_feature = self.FeatureExtraction(input)
		visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
		visual_feature = visual_feature.squeeze(3)
		
		
		# recognition stage :
		prediction = self.recogniser(visual_feature,text,is_train, self.opt.batch_max_length, regex,self.character)
		
		""" Sequence modeling stage """
		# BiLSTM encoder is not used in transformer.
		if self.stages['Seq'] == 'BiLSTM' and self.stages['Pred'] != "transformer":
			contextual_feature = self.SequenceModeling(visual_feature)
		else:
			contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

		""" Prediction stage """
		if self.stages['Pred'] == 'CTC':
			prediction = self.Prediction(contextual_feature.contiguous())
		if self.stages['Pred'] == "Attn":
			prediction = self.Prediction(contextual_feature.contiguous(), text, is_train,
			                             self.opt.batch_max_length, regex, self.character)
		else:
			src_mask, tgt_mask, tgt_padding_mask = create_mask(contextual_feature.contiguous(), text)
			
			prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, self.opt.batch_max_length,
			                             regex,
			                             self.character, tgt_mask, tgt_padding_mask)
		return prediction