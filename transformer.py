import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt

def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
    # [[False, False, False, False,  True],
    #  [False, False, False, False,  True],
    #  [False, False, False, False,  True],
    #  [False, False, False, False,  True],
    #  [False, False, False, False,  True]]
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

def get_attn_subsequence_mask(seq):
    #seq: [batch_size, tgt_len]
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1) # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    # [[0, 1, 1, 1, 1, 1],
    #  [0, 0, 1, 1, 1, 1],
    #  [0, 0, 0, 1, 1, 1],
    #  [0, 0, 0, 0, 1, 1],
    #  [0, 0, 0, 0, 0, 1],
    #  [0, 0, 0, 0, 0, 0]]
    return subsequence_mask.to(device) # [batch_size, tgt_len, tgt_len]

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9)  # 把被mask的地方置为无限小，softmax之后基本就是0，对q的单词不起作用

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn

class AddNorm(nn.Module):
    """残差连接后进行层规范化。"""
    def __init__(self, normalized_shape, dropout=0.1):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.layernorm(self.dropout(Y) + X)

class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self):
        super(MultiHeadAttention, self).__init__()

        assert (d_k * n_heads == d_model)  # 词向量维度应该能够被heads整除
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        batch_size = input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return output,attn

class PositionWiseFeedForwardNet(nn.Module):
    """基于位置的前馈网络"""
    def __init__(self):
        super(PositionWiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
    def forward(self, inputs):
        #inputs: [batch_size, seq_len, d_model]
        return self.fc(inputs) # [batch_size, seq_len, d_model]

class PositionalEncoding(nn.Module):
    """PositionalEncoding 代码实现"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # PE(pos,2i) = sin(pos / 10000^{2i/d_model})
        # PE(pos,2i+1) = cos(pos / 10000^{2i/d_model})
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.pow(10000,torch.arange(0, d_model, 2, dtype=torch.float) / d_model)
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)

        pe = pe.unsqueeze(0).transpose(0, 1) #(max_len*1*d_model)

        self.register_buffer('pe', pe)  ## 定一个缓冲区，其实简单理解为这个参数不更新就可以

    def forward(self, x):
        #x: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class EncoderBlock(nn.Module):
    """transformer编码器块。"""
    def __init__(self):
        super(EncoderBlock, self).__init__()
        self.enc_attention = MultiHeadAttention()
        self.addnorm1 = AddNorm(d_model)
        self.pos_ffn = PositionWiseFeedForwardNet()
        self.addnorm2 = AddNorm(d_model)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs1,attn = self.enc_attention(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs2 = self.addnorm1(enc_inputs, enc_outputs1)
        enc_outputs3 =  self.addnorm2(enc_outputs2, self.pos_ffn(enc_outputs2))
        return enc_outputs3,attn

class DecoderBlock(nn.Module):
    """transformer解码器块。"""
    def __init__(self):
        super(DecoderBlock, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.addnorm1 = AddNorm(d_model)
        self.dec_enc_attn = MultiHeadAttention()
        self.addnorm2 = AddNorm(d_model)
        self.pos_ffn = PositionWiseFeedForwardNet()
        self.addnorm3 = AddNorm(d_model)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        '''
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        '''
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs1, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs2 = self.addnorm1(dec_inputs, dec_outputs1)
        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs3, dec_enc_attn = self.dec_enc_attn(dec_outputs2, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs4 = self.addnorm2(dec_outputs2, dec_outputs3)
        dec_outputs5 = self.pos_ffn(dec_outputs4) # [batch_size, tgt_len, d_model]
        dec_outputs6 = self.addnorm3(dec_outputs4, dec_outputs5)
        return dec_outputs6, dec_self_attn, dec_enc_attn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderBlock() for _ in range(n_layers)])

    def forward(self,enc_inputs):
        #enc_inputs: [batch_size, src_len]
        enc_outputs = self.src_emb(enc_inputs)  # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)  # [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderBlock() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        '''
        dec_inputs: [batch_size, tgt_len]
        enc_intpus: [batch_size, src_len]
        enc_outputs: [batsh_size, src_len, d_model]
        '''
        dec_outputs = self.tgt_emb(dec_inputs) # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1) # [batch_size, tgt_len, d_model]
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs) # [batch_size, tgt_len, tgt_len]
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs) # [batch_size, tgt_len, tgt_len]
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0) # [batch_size, tgt_len, tgt_len]

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs) # [batc_size, tgt_len, src_len]

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self,enc_inputs, dec_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        '''
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        # dec_outpus: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs)  # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns

def make_data(sentences):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
        enc_input = [[src_vocab[n] for n in sentences[i][0].split()]]
        dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]]
        dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]]

        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)

    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)

class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]

def showgraph(attn):
    for i in range(n_batchs):
        fig = plt.figure(figsize=(10, 8))
        for j in range(n_heads):
            sub_attn = attn[-1][i,j,...]
            sub_attn = sub_attn.data.cpu().numpy()
            ax = fig.add_subplot(2,4,j+1)
            ax.matshow(sub_attn)

def greedy_decoder(model, enc_input, start_symbol):
    #Greedy Decoder is Beam search when K=1
    #TODO: Beam search
    enc_outputs, enc_self_attns = model.encoder(enc_input)
    dec_input = torch.zeros(1, 0).type_as(enc_input.data)
    terminal = False
    next_symbol = start_symbol
    while not terminal:
        dec_input=torch.cat([dec_input.detach(),torch.tensor([[next_symbol]],dtype=enc_input.dtype,device=device)],-1)
        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[-1]
        next_symbol = next_word
        if next_symbol == tgt_vocab["E"]:
            terminal = True
    return dec_input

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sentences = [
        # enc_input           dec_input         dec_output
        ['今 晚 的 夜 色 真 美 。', 'S 今 夜 の 夜 は 绮 丽 で す 。 E P P P', '今 夜 の 夜 は 绮 丽 で す 。 E P P P P'],
        ['今 晚 月 色 真 美 。 P', 'S 月 が 绮 丽 で す ね 。 E P P P P P', '月 が 绮 丽 で す ね 。 E P P P P P P'],
        ['今 晚 的 酒 真 香 。 P', 'S 今 晩 の 酒 は 本 当 に い い で す ね 。','今 晩 の 酒 は 本 当 に い い で す ね 。 E']
    ]

    src_vocab = {'P': 0, '今': 1, '晚': 2, '的': 3, '夜': 4, '色': 5, '真': 6, '美': 7, '。': 8, '月': 9, '酒': 10, '香': 11}
    src_vocab_size = len(src_vocab)
    src_idx2word = ['P', '今', '晚', '的', '夜', '色', '真', '美', '。', '月', '酒', '香']

    tgt_vocab = {'P': 0, 'S': 1, 'E': 2, '今': 3, '夜': 4, 'の': 5, 'は': 6, '绮': 7, '丽': 8, 'で': 9, 'す': 10, '。': 11, '月': 12, 'が': 13, 'ね': 14, '晩': 15, '酒': 16, '本': 17, '当': 18, 'に': 19, 'い': 20}
    tgt_vocab_size = len(tgt_vocab)
    tgt_idx2word = ['P', 'S', 'E', '今', '夜', 'の', 'は', '绮', '丽', 'で', 'す', '。', '月', 'が', 'ね', '晩', '酒', '本', '当', 'に', 'い']

    src_len = 8
    tgt_len = 15

    # Transformer Parameters
    d_model = 512  # Embedding Size
    d_ff = 2048  # FeedForward dimension
    d_k = d_v = 64  # dimension of K(=Q), V
    n_layers = 6  # number of Encoder of Decoder Layer
    n_heads = 8  # number of heads in Multi-Head Attention
    n_batchs = 3

    enc_inputs, dec_inputs, dec_outputs = make_data(sentences)
    loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), n_batchs, True)
    model = Transformer()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

    for epoch in range(1000):
        for enc_inputs, dec_inputs, dec_outputs in loader:
            enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
            # outputs: [batch_size * tgt_len, tgt_vocab_size]
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
            loss = criterion(outputs, dec_outputs.view(-1))
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    #show attn
    showgraph(enc_self_attns)
    showgraph(dec_self_attns)
    showgraph(dec_enc_attns)

    # Test
    enc_inputs, _, _ = next(iter(loader))
    for i in range(len(enc_inputs)):
        greedy_dec_input = greedy_decoder(model, enc_inputs[i].view(1, -1).to(device), start_symbol=tgt_vocab["S"])
        predict, _, _, _ = model(enc_inputs[i].view(1, -1).to(device), greedy_dec_input)
        predict = predict.data.max(1, keepdim=True)[1]
        print([src_idx2word[j] for j in enc_inputs[i]], '->', [tgt_idx2word[n.item()] for n in predict.squeeze()])
