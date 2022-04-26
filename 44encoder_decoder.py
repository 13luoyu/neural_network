# 编码器和解码器
#               CNN                         RNN
# 编码器：将输入编程成中间表达形式（特征）      将文本表示成向量
# 解码器：将中间表示解码成输出                将向量表示成输出
# Input -> Encoder -> State -> Decoder -> Output
#                     Input ->

from torch import nn


class Encoder(nn.Module):
    """编码器-解码器架构的基本编码器接口"""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError



class Decoder(nn.Module):
    """编码器-解码器架构的基本解码器接口"""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):  # 拿到编码器的输出进行初始化
        raise NotImplementedError

    def forward(self, X, state):  # 依据input和state解码
        raise NotImplementedError


#@save
class EncoderDecoder(nn.Module):
    """编码器-解码器架构的基类"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)  # 编码
        dec_state = self.decoder.init_state(enc_outputs, *args)  # 初始状态
        return self.decoder(dec_X, dec_state)  # 解码





