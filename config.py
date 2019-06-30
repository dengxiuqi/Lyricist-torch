clip_grad = 5.				# 防止梯度爆炸的截断梯度
decay_rate = 0.98			# learning_rate衰减率
decay_epoch = 2				# 衰减步长
fix_length = 32				# 每个序列的长度
model_path = "/model/"		# 模型保存/加载的路径

# network structure
encoder_layers = 2			# 编码器RNN层数
decoder_layers = 2			# 解码器RNN层数
embedding_dim = 200			# Embedding输出维度, word2vec维度
output_size = 10000			# 网络输出的维度
encoder_hidden_size = 200	# 编码器的hidden_layer大小
decoder_hidden_size = 200	# 解码器的hidden_layer大小
dropout = 0.1
