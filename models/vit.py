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
from einops import rearrange

from model_factories.RecogniserFactory import RecogniserFactory
from model_factories.ModelFactory import TransformationModelFactory, FeatureExtractorFactory
from modules.recognition.vision_transformer import VisionTransformerRecogniser


class Vit(nn.Module):
	"""
	This class represents generic text recognition model of 3 stages, 1) spatical transformation  2) feature
	extraction   3) Seq2seq processor like LSTM , CTC, Transformer etc.
	"""
	
	def __init__(self, opt, character):
		super(Vit, self).__init__()
		self.opt = opt
		
		# self.character is passed to prediction model to mask certain characters during prediction of regex based text
		self.character = character
		self.patch_size= opt.patch_size
		
		""" Transformation """
		self.Transformation = TransformationModelFactory.get_transformation_model(opt)
		
		"""Recogniser"""
		self.recogniser = RecogniserFactory.get_recogniser(opt)
	
	# regex is used if we expect predicted text to follow certain pattern. ex: regex for PAN number is "[A-Z]{5}[
	# 0-9]{4}[A-Z]{1}" so here for first five positions, predicted probablities of numbers and special characters
	# are set to -infinity.
	def forward(self, input, text, is_train=True, regex=None, character_list=None):
		
		if self.Transformation is not None:
			input = self.Transformation(input)
		
		if isinstance(self.recogniser, VisionTransformerRecogniser):
			prediction = self.recogniser(input, self.opt.batch_max_length+1)
		
		else:
			prediction=self.recogniser(input, text, is_train, self.opt.batch_max_length , regex,character_list)
		return prediction
