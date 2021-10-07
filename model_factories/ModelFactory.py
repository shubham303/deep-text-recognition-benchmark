class TransformationModelFactory():
	def __init__(self):
		super(TransformationModelFactory, self).__init__()
	
	@staticmethod
	def get_transformation_model(self,opt):
		if opt.Transformation ==None:
			print("no transformation module specified")
			return None
		
		if opt.Transformation == "TPS":
			return TPS_SpatialTransformerNetwork(
				F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW),
				I_channel_num=opt.input_channel)
		
		else:
			print("{} transformation module not defined",format(opt.Transformation))
			
