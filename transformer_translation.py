import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import jieba
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import collections
import os
import torch.nn.functional as F
from fastai import losses

class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing: float = 0.1,
                 reduction="mean", weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing   = smoothing
        self.reduction = reduction
        self.weight    = weight

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
         if self.reduction == 'sum' else loss

    def linear_combination(self, x, y):
        return self.smoothing * x + (1 - self.smoothing) * y

    def forward(self, preds, target):
        assert 0 <= self.smoothing < 1

        if self.weight is not None:
            self.weight = self.weight.to(preds.device)

        n = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1))
        nll = F.nll_loss(
            log_preds, target, reduction=self.reduction, weight=self.weight
        )
        return self.linear_combination(loss / n, nll)

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
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
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

class MyDataSet(Data.Dataset):
    def __init__(self,en_datas,ch_datas,en_word2index,ch_word2index):
        self.en_datas = en_datas
        self.ch_datas = ch_datas
        self.en_word2index = en_word2index
        self.ch_word2index = ch_word2index

    def __getitem__(self,index):
        en_sentence = self.en_datas[index]
        ch_sentence = self.ch_datas[index]

        en_list = en_sentence.split(" ")
        ch_list = ch_sentence.split(" ")

        en_index = [self.en_word2index[i] for i in en_list]
        ch_index = [self.ch_word2index[i] for i in ch_list]

        return en_index,ch_index

    def __len__(self):
        assert len(self.en_datas) == len(self.ch_datas)
        return len(self.en_datas)

def my_collate(batch_datas):
    en_datas,ch_datas = [],[]
    en_lens,ch_lens = [],[]
    for en_index,ch_index in batch_datas:
        en_datas.append(en_index)
        ch_datas.append(ch_index)
        en_lens.append(len(en_index))
        ch_lens.append(len(ch_index))

    max_en_len = max(en_lens)
    max_ch_len = max(ch_lens)

    en_indexs = [i+[en_word2index["<PAD>"]] * (max_en_len - len(i)) for i in en_datas]
    ch_indexs = [[ch_word2index["<BOS>"]]  + i + [ch_word2index["<EOS>"]] + [ch_word2index["<PAD>"]] * (max_ch_len - len(i))  for i in ch_datas]

    return torch.LongTensor(en_indexs), torch.LongTensor(ch_indexs)[:,:-1], torch.LongTensor(ch_indexs)[:,1:]

def get_datas(file):
    all_data = pd.read_csv(file)
    return all_data["english"].tolist(), all_data["chinese"].tolist()

def get_en_txt(filename,textlist):
    resultlist = []
    with open(filename,'w',encoding="utf-8") as f:
        for text in textlist:
            newtext = ""
            punctuation = [".",",","!","?"]
            for word in text:
                if word in punctuation:
                    word= " " + word
                newtext += word
            resultlist.append(newtext)
            f.write(newtext+"\n")
    return resultlist

def get_ch_txt(filename,textlist):
    resultlist = []
    for setence in textlist:
        result = [i for i in jieba.lcut(setence) if i !=" "]
        text = " ".join(result)
        resultlist.append(text)
    with open(filename,'w',encoding="utf-8") as f:
        for i in resultlist:
            f.write(i+"\n")
    return resultlist

def get_vec(file_name):
    sentences = LineSentence(file_name)
    model = Word2Vec(sentences,sg=1,hs=0,vector_size=50,window=3,min_count=1,workers=10)
    word_2_index = model.wv.key_to_index
    index_2_word = model.wv.index_to_key
    return word_2_index,index_2_word

def bleu(pred_seq, label_seq, k=2):
    """计算BLEU"""
    len_pred, len_label = len(pred_seq), len(label_seq)
    if len_pred == 0:
        return 0.0
    else:
        score = np.exp(min(0, 1 - len_label / len_pred))
        for n in range(1, k + 1):
            num_matches, label_subs = 0, collections.defaultdict(int)
            for i in range(len_label - n + 1):
                label_subs[''.join(label_seq[i: i + n])] += 1
            if (len_pred - n + 1):
                for i in range(len_pred - n + 1):
                    if label_subs[''.join(pred_seq[i: i + n])] > 0:
                        num_matches += 1
                        label_subs[''.join(pred_seq[i: i + n])] -= 1
                score *= np.power(num_matches / (len_pred - n + 1), np.power(0.5, n))
        return score

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
        if next_symbol == tgt_vocab["<EOS>"] or dec_input.shape[1] > 50:
            terminal = True
    return dec_input

def beam_search(model,enc_input,label,totalscore,num_beams = 3):
    enc_outputs, _ = model.encoder(enc_input)
    w_index = tgt_vocab["<BOS>"]
    beams = [[[w_index], 0.0]]
    temp = []
    result = []
    pred_list = []
    max_len = 50
    for _ in range(max_len):
        for beam in beams:
            dec_input = torch.LongTensor(beam[0]).unsqueeze(0).to(device)
            dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
            pre = model.projection(dec_outputs)
            pre = torch.softmax(pre,dim=-1)
            top_k_probs, w_indexs = torch.topk(pre, dim=-1, k=num_beams)
            top_k_log_probs = torch.log(top_k_probs)
            top_k_log_probs = top_k_log_probs.squeeze(0)[-1]
            w_indexs = w_indexs.squeeze(0)[-1]
            for top_k_log_prob,w_index2 in zip(top_k_log_probs,w_indexs):
                gen_seq = beam[0] + [w_index2.item()]
                prob = beam[1] + top_k_log_prob.item()
                temp.append([gen_seq,prob])
        beams = sorted(temp,key= lambda x:x[1],reverse=True)
        if len(beams) > num_beams:
            beams = beams[:num_beams]
        temp = beams
        for beam in beams:
            if beam[0][-1] == tgt_vocab["<EOS>"]:
                result.append(beam)
                temp.remove(beam)
        beams = temp
        temp = []
        if len(result) >= num_beams:
            break
    result = sorted(result,key=lambda x:x[1],reverse=True)
    for i in result[0][0][1:-1]:
        pred_list.append(ch_index2word[i])
    en_list = [src_idx2word[s] for s in enc_input[0]]
    ch_list = [tgt_idx2word[t] for t in label][1:]
    bleuscore = bleu(''.join(pred_list), ''.join(ch_list))
    totalscore += bleuscore
    print(f"原文:{' '.join(en_list)},译文:{''.join(ch_list)},AI翻译:{''.join(pred_list)},bleu:{bleuscore}")
    return totalscore

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    english_list,chinese_list = get_datas("translate.csv")
    en_datas_orign = get_en_txt("english.txt",english_list)
    ch_datas_orign = get_ch_txt("chinese.txt",chinese_list)
    en_word2index,en_index2word = get_vec("english.txt")
    ch_word2index,ch_index2word = get_vec("chinese.txt")

    # Transformer Parameters
    d_model = 512  # Embedding Size
    d_ff = 2048  # FeedForward dimension
    d_k = d_v = 64  # dimension of K(=Q), V
    n_layers = 6  # number of Encoder of Decoder Layer
    n_heads = 8  # number of heads in Multi-Head Attention
    n_batchs = 4
    epoch = 20

    ch_word2index = {k:v+3 for k,v in ch_word2index.items()}
    en_word2index = {k:v+1 for k,v in en_word2index.items()}

    ch_word2index.update({"<PAD>":0,"<BOS>":1,"<EOS>":2})
    en_word2index.update({"<PAD>":0})

    ch_index2word = ["<PAD>", "<BOS>", "<EOS>"] + ch_index2word
    en_index2word = ["<PAD>"] + en_index2word

    src_vocab = en_word2index
    tgt_vocab = ch_word2index
    src_idx2word = en_index2word
    tgt_idx2word = ch_index2word
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    en_datas = en_datas_orign[:20000]
    ch_datas = ch_datas_orign[:20000]
    test_en_datas = en_datas_orign[20000:]
    test_ch_datas = ch_datas_orign[20000:]

    loader = Data.DataLoader(MyDataSet(en_datas, ch_datas, src_vocab, tgt_vocab), n_batchs, True, collate_fn= my_collate)
    testloader = Data.DataLoader(MyDataSet(test_en_datas, test_ch_datas, src_vocab, tgt_vocab), 1, False, collate_fn= my_collate)
    model = Transformer()
    model = model.to(device)

    #criterion = nn.CrossEntropyLoss(ignore_index=0)
    criterion = LabelSmoothingLoss()
    #criterion = losses.LabelSmoothingCrossEntropy()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.99,weight_decay=1e-5)

    baseloss = 1.0
    if os.path.exists("mytransformer.pth"):
        model.load_state_dict(torch.load("mytransformer.pth"))
    else:
        for e in range(epoch):
            for enc_inputs, dec_inputs, dec_outputs in loader:
                enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
                # outputs: [batch_size * tgt_len, tgt_vocab_size]
                outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
                loss = criterion(outputs, dec_outputs.reshape(-1))
                print('Epoch:', '%04d' % (e + 1), 'loss =', '{:.6f}'.format(loss))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if loss < baseloss:
                    torch.save(model.state_dict(),"mytransformer.pth")
                    baseloss = loss
        model.load_state_dict(torch.load("mytransformer.pth"))
    total_score = 0.0
    #Test
    # for test_inputs, label, _ in testloader:
    #     test_inputs = test_inputs[0]
    #     label = label[0]
    #     greedy_dec_input = greedy_decoder(model, test_inputs.view(1, -1).to(device), start_symbol=tgt_vocab["<BOS>"])
    #     predict, _, _, _ = model(test_inputs.view(1, -1).to(device), greedy_dec_input)
    #     predict = np.argmax(predict.cpu().detach().numpy(),axis=1)
    #     en_list = [src_idx2word[s] for s in test_inputs]
    #     ch_list = [tgt_idx2word[t] for t in label][1:]
    #     result_list = [tgt_idx2word[n.item()] for n in predict][:-1]
    #     bleu_score = bleu(''.join(result_list), ''.join(ch_list))
    #     total_score += bleu_score
    #     print(f"原文:{' '.join(en_list)},译文:{''.join(ch_list)},AI翻译:{''.join(result_list)},bleu:{bleu_score}")
    # BeamSearch
    for test_inputs, label, _ in testloader:
        label = label[0]
        total_score = beam_search(model, test_inputs.to(device),label,total_score)
    print(total_score/len(testloader))