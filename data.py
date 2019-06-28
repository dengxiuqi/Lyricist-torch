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
        :param batch_size: 每个batch的大小. 默认: 128
        :param fix_length: 每个序列的最大长度, 长度不足的句子会用"<pad>"补齐, 超过的句子会被截断. 默认: 32
        :param singer: 为None时读取所有歌曲; 否则只读取对应歌手的歌曲. 默认: None
        :param target_vocab_size: 目标词袋的长度, 在输出端(目标)只保留词频最高的前 target_vocab_size 个词语,
                            其它词语都会被"<unk>"替换. 默认: 5000
        :param vector_path: word2vec模型的路径. PS: 必须是.txt格式的文件
        :param device: 设备, "cuda"或"cpu". 默认: None, 自动选择"cuda"或"cpu"
        """
        self.batch_size = batch_size
        self.fix_length = fix_length
        self.singer = singer
        self.target_vocab_size = target_vocab_size
        self.vector_path = vector_path
        self._proprecess()
        self.DEVICE = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenize = lambda x: jieba.lcut(x, HMM=False)
        self.ENCODER = Field(sequential=True, tokenize=self.tokenize, batch_first=True, fix_length=self.fix_length,
                             include_lengths=True, init_token="<go>", eos_token="<eos>")
        self.DECODER = Field(sequential=True, tokenize=self.tokenize, batch_first=True, fix_length=self.fix_length,
                             include_lengths=True, init_token="<go>", eos_token="<eos>")
        self.TARGET = Field(sequential=True, tokenize=self.tokenize, batch_first=True, fix_length=32, eos_token="<eos>")
        self.dataset = self._build_dataset()
        self.vectors = Vectors(name=self.vector_path, cache=FILE_PATH + "/temp")
        self._build_vocab()
        self._build_vector()
        self.stoi = self.ENCODER.vocab.stoi     # 从词语到id的映射字典
        self.itos = self.ENCODER.vocab.itos     # 从id到词典的映射字典
        self.vocab_size = len(self.ENCODER.vocab)           # 词典的大小
        self.vector_dim = self.vectors.dim                  # 词向量的维度
        self.vector_weights = self.ENCODER.vocab.vectors    # 词向量的权重
        self.target_vocab_size = len(self.TARGET.vocab)
        self.data_iter = BucketIterator(self.dataset, batch_size=self.batch_size, shuffle=True, device=self.DEVICE)

    def process(self, text, return_length=False):
        encoder_input, encoder_length = self.ENCODER.process([self.ENCODER.preprocess(text)])
        if return_length:
            return encoder_input, encoder_length
        else:
            return encoder_input

    def logist2word(self, logist, topn=1):
        ids = logist.view(-1).argsort(descending=True)
        word = [self.itos[i] for i in ids[:topn]]
        return word

    def _build_dataset(self):
        """
        读取数据
        """
        fields = {'encoder': ('encoder', self.ENCODER),
                  'decoder': ('decoder', self.DECODER),
                  'target': ('target', self.TARGET)}
        dataset = TabularDataset(path=FILE_PATH+"/temp/data.json", format="json", fields=fields)
        return dataset

    def _build_vocab(self):
        """
        构建词典
        """
        self.ENCODER.build_vocab(self.dataset, max_size=self.target_vocab_size)
        self.DECODER.build_vocab()
        self.TARGET.build_vocab()

        self.TARGET.vocab.stoi = self.ENCODER.vocab.stoi
        self.TARGET.vocab.itos = self.ENCODER.vocab.itos

        self.ENCODER.build_vocab(self.dataset, vectors=self.vectors)
        self.DECODER.vocab.stoi = self.ENCODER.vocab.stoi
        self.DECODER.vocab.itos = self.ENCODER.vocab.itos
        self.ENCODER.init_token = None
        self.ENCODER.eos_token = None

    def _build_vector(self):
        if not os.path.exists(FILE_PATH+"/temp"):    # 如果./temp文件不存在, 就创建
            os.mkdir(FILE_PATH+"/temp")
        self.ENCODER.vocab.set_vectors(self.vectors.stoi, self.vectors.vectors, self.vectors.dim)

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
                    encoder = song["lyric"][i]
                    decoder = song["lyric"][i+1]
                    target = song["lyric"][i+1]
                    data.append({"encoder": encoder, "decoder": decoder, "target": target})

        if not os.path.exists(FILE_PATH+"/temp"):    # 如果./temp文件不存在, 就创建
            os.mkdir(FILE_PATH+"/temp")

        with open(FILE_PATH+"/temp/data.json", "w") as f:  # 保存为临时文件, 以便torchtext读取
            f.writelines([json.dumps(d, ensure_ascii=False) + "\n" for d in data])


class Lyric(Data):
    def __init__(self, batch_size=128, fix_length=32, singer=None, target_vocab_size=5000, vector_path=VEC_PATH, device=None):
        super(Lyric, self).__init__(batch_size=batch_size,
                                    fix_length=fix_length,
                                    singer=None,
                                    target_vocab_size=target_vocab_size,
                                    vector_path=vector_path,
                                    device=device)
        if singer is not None:
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

            self.ENCODER.vocab.stoi = self.ENCODER_stoi
            self.ENCODER.vocab.itos = self.ENCODER_itos
            self.DECODER.vocab.stoi = self.DECODER_stoi
            self.DECODER.vocab.itos = self.DECODER_itos
            self.TARGET.vocab.stoi = self.TARGET_stoi
            self.TARGET.vocab.itos = self.TARGET_itos
            self._build_vector()
            self.stoi = self.ENCODER.vocab.stoi  # 从词语到id的映射字典
            self.itos = self.ENCODER.vocab.itos  # 从id到词典的映射字典
            self.vocab_size = len(self.ENCODER.vocab)  # 词典的大小
            self.vector_dim = self.vectors.dim  # 词向量的维度
            self.vector_weights = self.ENCODER.vocab.vectors  # 词向量的权重
            self.target_vocab_size = len(self.TARGET.vocab)
            self.data_iter = BucketIterator(self.dataset, batch_size=self.batch_size, shuffle=True, device=self.DEVICE)
