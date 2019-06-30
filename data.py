from torchtext.data import Field, TabularDataset, BucketIterator
from torchtext.vocab import Vectors
import torch
import json
import jieba
import os
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
VEC_PATH = FILE_PATH+'/data/word2vec.txt'


class Data(object):
    def __init__(self, batch_size=128, fix_length=32, singer=None, target_vocab_size=5000, vector_path=VEC_PATH, device=None):
        """
        用于生成歌词生成任务的数据预处理和Batch生成
        每次输入网络的数据包括:
            encoder_input:  编码器输入, shape: (batch_size, time_step, word_id)
            encoder_length: 编码器输入文本有效长度, shape: (batch_size, )
            decoder_input:  解码器输入, shape: (batch_size, time_step, word_id)
            decoder_length: 解码器输入文本有效长度, shape: (batch_size, )
            target: 解码器输出目标, 用于计算Loss, shape: (batch_size, time_step, word_id)
        :param batch_size: 每个batch的大小. 默认: 128
        :param fix_length: 每个序列的最大长度, 长度不足的句子会用"<pad>"补齐, 超过的句子会被截断. 默认: 32
        :param singer: 为None时读取所有歌曲; 否则只读取对应歌手的歌曲. 默认: None
        :param target_vocab_size: 目标词典(解码器输出)的长度, 在输出端(目标)只保留词频最高的前 target_vocab_size 个词语,
                            其它词语都会被"<unk>"替换. 默认: 5000
        :param vector_path: word2vec模型的路径. PS: 必须是.txt格式的文件
        :param device: 设备, "cuda"或"cpu". 默认: None, 自动选择"cuda"或"cpu"
        """
        self.batch_size = batch_size
        self.fix_length = fix_length
        self.singer = singer
        self.target_vocab_size = target_vocab_size
        self.vector_path = vector_path
        self.DEVICE = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenize = lambda x: jieba.lcut(x, HMM=False)  # 分词
        
        # 定义torchtext的三个用于文本预处理的Field对象, 其中ENCODER并不需要句首符"<go>"和末尾符"<eos>", 
        # 但为了三个Field对象对文本编码解码的一致性, 在定义ENCODER Field对象时要将它们声明, 
        # 在词典映射构建完毕后再将它们去掉, 在self._build_vocab 中也有说明
        self.ENCODER = Field(sequential=True, 
                             tokenize=self.tokenize, 
                             batch_first=True,              # 数据的第一维是batch(默认是time_step)
                             fix_length=self.fix_length,    # 固定句子长度, 长度不足的句子会用"<pad>"补齐, 超过的句子会被截断
                             include_lengths=True,          # 处理文本时除了返回编码后的文本, 同时返回文本的长度
                             init_token="<go>",             # 文本的句首会自动添加"<go>"
                             eos_token="<eos>")             # 文本的末尾会自动添加"<eos>"
        self.DECODER = Field(sequential=True, 
                             tokenize=self.tokenize, 
                             batch_first=True, 
                             fix_length=self.fix_length,
                             include_lengths=True, 
                             init_token="<go>", 
                             eos_token="<eos>")
        self.TARGET = Field(sequential=True, 
                            tokenize=self.tokenize, 
                            batch_first=True, 
                            fix_length=self.fix_length, 
                            eos_token="<eos>")              # 由于`target`是`decoder`左移一位的结果, 所以不需要句首符"<go>"

        # 数据处理
        self._proprecess()                                  # 对语料库进行读取, 并转化维torchtext能识别的.json文件格式
        self.dataset = self._build_dataset()                # 读取处理后的数据, 生成torchtext的DataSet对象
        self.vectors = Vectors(name=self.vector_path, cache=FILE_PATH + "/temp")    # 加载word2vec词向量
        self._build_vocab()                                 # 构建词典映射
        self._build_vector()                                # 构建词向量映射
        self.stoi = self.ENCODER.vocab.stoi                 # 从词语到id的映射字典
        self.itos = self.ENCODER.vocab.itos                 # 从id到词典的映射字典
        self.vocab_size = len(self.ENCODER.vocab)           # 词典的大小
        self.vector_dim = self.vectors.dim                  # 词向量的维度
        self.vector_weights = self.ENCODER.vocab.vectors    # 词向量的权重
        self.target_vocab_size = len(self.TARGET.vocab)     # 重新赋值, 因为加入了"<eos>"等标志位的实际词典大会大于原target_vocab_size

        # 迭代器, 用于训练时生成batch
        self.data_iter = BucketIterator(self.dataset,
                                        batch_size=self.batch_size,
                                        shuffle=True,       # 打乱数据原本顺序
                                        device=self.DEVICE)

    def process(self, text, return_length=False, go=None, eos=None):
        """
        文本数据预处理(编码), 用于生成可以输入pytorch神经网络的Tensor格式数据
        :param text: 原始文本, str格式
        :param return_length: 是否返回句子长度, 默认: False
        :param go: 是否添加句首符"<go>"
        :param eos: 是否添加句末符"<eos>"
        :return: Tensor格式, 编码后的文本(和句子长度)
        """
        tokens = self.ENCODER.preprocess(text)      # 用ENCODER Field对text进行分词
        if go:
            tokens.insert(0, "<go>")                # 在句首添加"<go>"
        if eos:
            tokens.append("<eos>")                  # 在句末添加"<eos>"
        encoder_input, encoder_length = self.ENCODER.process([tokens])  # 将句子编码
        encoder_input = encoder_input.to(self.DEVICE)
        encoder_length = encoder_length.to(self.DEVICE)
        if return_length:
            return encoder_input, encoder_length
        else:
            return encoder_input

    def logist2word(self, logist, topn=1):
        """
        将pytorch神经网络输出的logist转换为对应的文本, 用于在test阶段处理batch_size=1的数据
        :param logist: 神经网络输出, shape: [1, self.target_vocab_size]
        :param topn: 保留概率最大的前topn个词语, 默认为1
        :return:
        """
        ids = logist.view(-1).argsort(descending=True)  # 将下标按输出值进行排序
        word = [self.itos[i] for i in ids[:topn]]       # 将下标id转为对应词语
        return word

    def _build_dataset(self):
        """
        读取由self._proprecess方法处理后的数据, 生成torchtext的DataSet对象
        """
        fields = {'encoder': ('encoder', self.ENCODER),
                  'decoder': ('decoder', self.DECODER),
                  'target': ('target', self.TARGET)}
        dataset = TabularDataset(path=FILE_PATH+"/temp/data.json", format="json", fields=fields)
        return dataset

    def _build_vocab(self):
        """
        为之前定义的三个Field对象构建词典映射.
        由于`encoder`/`decoder`/`target`都涉及到对文本的编码解码, 但语料库却不完全一致, 若分别由前面定义的
        三个Field对象分别处理对应的部分, 那么对同一个词的编码会在decoder和encoder端出现不一致, 例如`晴天`在
        ENCODER词典中的id是8764, 在DECODER中的id是8892. 这样在网络的ENCODER和DECODER端要分别使用两个不同
        的Embedding层, 增加了网络参数量.
        为了保证词表的一致性, 我们在实例化ENCODER对象的时候也一并声明了其并不需要的句首符"<go>"和末尾符"<eos>".
        因此在构建词典时我们全部用ENCODER构建, 然后再将其编码解码映射表(itos和stoi)赋予DECODER和TARGET.
        """
        self.ENCODER.build_vocab(self.dataset,
                                 max_size=self.target_vocab_size)   # 定义max_size是因为本次构建的词表是给TARGET使用的
        self.DECODER.build_vocab()
        self.TARGET.build_vocab()

        self.TARGET.vocab.stoi = self.ENCODER.vocab.stoi    # 将word到id的词典赋予TARGET
        self.TARGET.vocab.itos = self.ENCODER.vocab.itos    # 将id到word的词典赋予TARGET

        self.ENCODER.build_vocab(self.dataset,              # 再次构建词典, 这次不带max_size参数
                                 vectors=self.vectors)      # 加上词向量参数, 因为网络的输入要涉及到word->id->vector的映射
        self.DECODER.vocab.stoi = self.ENCODER.vocab.stoi   # 将word到id的词典赋予DECODER
        self.DECODER.vocab.itos = self.ENCODER.vocab.itos   # 将id到word的词典赋予DECODER
        self.ENCODER.init_token = None                      # 清除ENCODER的句首符"<go>"
        self.ENCODER.eos_token = None                       # 清除ENCODER的末尾符"<eos>"

    def _build_vector(self):
        """
        构建词向量映射, 将word2vec中的词向量按ENCODER和DECODER词典id顺序排列, 用于网络的Embedding层
        """
        if not os.path.exists(FILE_PATH+"/temp"):           # 如果./temp文件不存在, 就创建
            os.mkdir(FILE_PATH+"/temp")                     # 将word2vec的缓存文件放在temp文件夹
        self.ENCODER.vocab.set_vectors(self.vectors.stoi,
                                       self.vectors.vectors,
                                       self.vectors.dim)

    def _proprecess(self):
        """
        对语料库进行读取, 并转化维torchtext能识别的.json文件格式, 形如:
            {"encoder": "鱼", "decoder": "我坐在椅子上看日出复活", "target": "我坐在椅子上看日出复活"}
            {"encoder": "我坐在椅子上看日出复活", "decoder": "我坐在夕阳里看城市的衰弱", "target": "我坐在夕阳里看城市的衰弱"}
            {"encoder": "我坐在夕阳里看城市的衰弱", "decoder": "我摘下一片叶子让它代替我", "target": "我摘下一片叶子让它代替我"}
            ...
        """
        data = []
        with open(FILE_PATH+"/data/songs.json") as f:
            songs = json.loads(f.read())
        for song in songs:
            if not self.singer or song["singer"] == self.singer:    # 按指定歌手进行读取
                # (歌名，首句) 组成第一条数据
                data.append({"encoder": song["title"], "decoder": song["lyric"][0], "target": song["lyric"][0]})
                for i in range(len(song["lyric"])-1):
                    # (前句，后句) 组成一条数组
                    encoder = song["lyric"][i]      # encoder输入的文本, 上句
                    decoder = song["lyric"][i+1]    # decoder输入的文本, 下句
                    target = song["lyric"][i+1]     # decoder输出的目标, 下句左移一个字
                    data.append({"encoder": encoder, "decoder": decoder, "target": target})

        if not os.path.exists(FILE_PATH+"/temp"):    # 如果./temp文件不存在, 就创建
            os.mkdir(FILE_PATH+"/temp")

        with open(FILE_PATH+"/temp/data.json", "w") as f:  # 保存为临时文件, 以便torchtext读取
            f.writelines([json.dumps(d, ensure_ascii=False) + "\n" for d in data])


class Lyric(Data):
    def __init__(self, batch_size=128, fix_length=32, singer=None, target_vocab_size=5000, vector_path=VEC_PATH, device=None):
        """
        继承Data类, 在pre_train时我们加载整个语料库进行训练, 在fine tune时我们只加载特定歌手的数据,
        如果直接调用Data类, 因为语料库不同, 在这两个阶段对同一个词语会编码得到不同的word_id, 导致Embedding
        层映射错误. 因此如果只加载某位歌手的数据, 我们同样先加载整个语料库并构建词典, 将词典临时保存,
        然后再加载特定歌手的语料, 再用临时保存的词典覆盖掉新生成的词典
        这样, 每个Batch生成的数据均来自指定歌手, 但词典映射仍与加载整个语料库时无异, 实现了网络的复用

        用于生成歌词生成任务的数据预处理和Batch生成
        每次输入网络的数据包括:
            encoder_input:  编码器输入, shape: (batch_size, time_step, word_id)
            encoder_length: 编码器输入文本有效长度, shape: (batch_size, )
            decoder_input:  解码器输入, shape: (batch_size, time_step, word_id)
            decoder_length: 解码器输入文本有效长度, shape: (batch_size, )
            target: 解码器输出目标, 用于计算Loss, shape: (batch_size, time_step, word_id)
        :param batch_size: 每个batch的大小. 默认: 128
        :param fix_length: 每个序列的最大长度, 长度不足的句子会用"<pad>"补齐, 超过的句子会被截断. 默认: 32
        :param singer: 为None时读取所有歌曲; 否则只读取对应歌手的歌曲. 默认: None
        :param target_vocab_size: 目标词典的长度, 在输出端(目标)只保留词频最高的前 target_vocab_size 个词语,
                            其它词语都会被"<unk>"替换. 默认: 5000
        :param vector_path: word2vec模型的路径. PS: 必须是.txt格式的文件
        :param device: 设备, "cuda"或"cpu". 默认: None, 自动选择"cuda"或"cpu"
        """
        super(Lyric, self).__init__(batch_size=batch_size,
                                    fix_length=fix_length,
                                    singer=None,
                                    target_vocab_size=target_vocab_size,
                                    vector_path=vector_path,
                                    device=device)
        if singer is not None:
            # 临时保存词典
            self.ENCODER_stoi = self.ENCODER.vocab.stoi
            self.ENCODER_itos = self.ENCODER.vocab.itos
            self.DECODER_stoi = self.DECODER.vocab.stoi
            self.DECODER_itos = self.DECODER.vocab.itos
            self.TARGET_stoi = self.TARGET.vocab.stoi
            self.TARGET_itos = self.TARGET.vocab.itos
            super(Lyric, self).__init__(batch_size=batch_size,
                                        fix_length=fix_length,
                                        singer=singer,
                                        target_vocab_size=target_vocab_size,
                                        vector_path=vector_path,
                                        device=device)
            # 覆盖新词典
            self.ENCODER.vocab.stoi = self.ENCODER_stoi
            self.ENCODER.vocab.itos = self.ENCODER_itos
            self.DECODER.vocab.stoi = self.DECODER_stoi
            self.DECODER.vocab.itos = self.DECODER_itos
            self.TARGET.vocab.stoi = self.TARGET_stoi
            self.TARGET.vocab.itos = self.TARGET_itos

            self._build_vector()
            self.stoi = self.ENCODER.vocab.stoi
            self.itos = self.ENCODER.vocab.itos
            self.vocab_size = len(self.ENCODER.vocab)
            self.vector_dim = self.vectors.dim
            self.vector_weights = self.ENCODER.vocab.vectors
            self.target_vocab_size = len(self.TARGET.vocab)
            self.data_iter = BucketIterator(self.dataset, batch_size=self.batch_size, shuffle=True, device=self.DEVICE)
