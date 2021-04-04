# -*- coding: utf-8 -*-

from django.shortcuts import render
from rest_framework import viewsets
from rest_framework.views import APIView
from .models import *
from .serializer import *
import json
from django.http.response import JsonResponse
from django.views.decorators.csrf import csrf_protect
from rest_framework.response import Response
from django.http import Http404
import torch
import torch.nn as nn
import torch.optim as optim
import csv

def read_csv(file):
    data = {}
    with open(file,'r', encoding="utf_8") as f:
        csv_data = csv.reader(f)
        for row in csv_data:
            if not data:
                data = row
    return data

class AIchatList(APIView):
#########################################
    queryset = AIchat.objects.all()
    serializer_class = AIchatSerializer
###########################################

###############################################################
    def get(self, request, format=None):
        snippets = AIchat.objects.all()
        serializer = AIchatSerializer(snippets, many=True)
        print(request.data)
        return Response(serializer.data)
        
###########################################################
    def post(self, request, format=None):
        data = None
        input_test_x = ''
        try:
            data = json.loads(request.body)
        except ValueError:
            pass

        if 'text' in request.GET:
            input_test_x = request.GET['text']

        elif data:
            input_test_x = data['text']


        csv_char = []
        char2id = {}

        csv_char = read_csv('./static/char2id.csv')
        id_cnt = 0
        for i in csv_char:
            char2id.update({i:id_cnt})
            id_cnt += 1

        test_x = []

        for d in input_test_x:
            test_x.append(char2id[d])
        test_x = [test_x]

        # 諸々のパラメータなど
        embedding_dim = 200
        hidden_dim = 128
        BATCH_NUM = 1
        vocab_size = len(char2id)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Encoderクラス
        class Encoder(nn.Module):
            def __init__(self, vocab_size, embedding_dim, hidden_dim):
                super(Encoder, self).__init__()
                self.hidden_dim = hidden_dim
                self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=char2id["　"])
                self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

            def forward(self, sequence):
                embedding = self.word_embeddings(sequence)
                # hsが各系列のGRUの隠れ層のベクトル
                # Attentionされる要素
                hs, h = self.gru(embedding)
                return hs, h

        # Attention Decoderクラス
        class AttentionDecoder(nn.Module):
            def __init__(self, vocab_size, embedding_dim, hidden_dim, batch_size):
                super(AttentionDecoder, self).__init__()
                self.hidden_dim = hidden_dim
                self.batch_size = batch_size
                self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=char2id["　"])
                self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
                # hidden_dim*2としているのは、各系列のGRUの隠れ層とAttention層で計算したコンテキストベクトルをtorch.catでつなぎ合わせることで長さが２倍になるため
                self.hidden2linear = nn.Linear(hidden_dim * 2, vocab_size)
                # 列方向を確率変換したいのでdim=1
                self.softmax = nn.Softmax(dim=1)

            def forward(self, sequence, hs, h):
                embedding = self.word_embeddings(sequence)
                output, state = self.gru(embedding, h)
            # Attention層
            # hs.size() = ([100, 29, 128])
            # output.size() = ([100, 10, 128])

            # bmmを使ってEncoder側の出力(hs)とDecoder側の出力(output)をbatchごとまとめて行列計算するために、Decoder側のoutputをbatchを固定して転置行列を取る
                t_output = torch.transpose(output, 1, 2) # t_output.size() = ([100, 128, 10])

                # bmmでバッチも考慮してまとめて行列計算
                s = torch.bmm(hs, t_output) # s.size() = ([100, 29, 10])

                # 列方向(dim=1)でsoftmaxをとって確率表現に変換
                # この値を後のAttentionの可視化などにも使うため、returnで返しておく
                attention_weight = self.softmax(s) # attention_weight.size() = ([100, 29, 10])

                # コンテキストベクトルをまとめるために入れ物を用意
                c = torch.zeros(self.batch_size, 1, self.hidden_dim, device=device) # c.size() = ([100, 1, 128])

                # 各DecoderのGRU層に対するコンテキストベクトルをまとめて計算する方法がわからなかったので、
                # 各層（Decoder側のGRU層は生成文字列が10文字なので10個ある）におけるattention weightを取り出してforループ内でコンテキストベクトルを１つずつ作成する
                # バッチ方向はまとめて計算できたのでバッチはそのまま
                for i in range(attention_weight.size()[2]): # 10回ループ

                    # attention_weight[:,:,i].size() = ([100, 29])
                    # i番目のGRU層に対するattention weightを取り出すが、テンソルのサイズをhsと揃えるためにunsqueezeする
                    unsq_weight = attention_weight[:,:,i].unsqueeze(2) # unsq_weight.size() = ([100, 29, 1])

                    # hsの各ベクトルをattention weightで重み付けする
                    weighted_hs = hs * unsq_weight # weighted_hs.size() = ([100, 29, 128])

                    # attention weightで重み付けされた各hsのベクトルをすべて足し合わせてコンテキストベクトルを作成
                    weight_sum = torch.sum(weighted_hs, axis=1).unsqueeze(1) # weight_sum.size() = ([100, 1, 128])

                    c = torch.cat([c, weight_sum], dim=1) # c.size() = ([100, i, 128])

                # 箱として用意したzero要素が残っているのでスライスして削除
                c = c[:,1:,:]

                output = torch.cat([output, c], dim=2) # output.size() = ([100, 10, 256])
                output = self.hidden2linear(output)
                return output, state, attention_weight

        encoder = Encoder(vocab_size, embedding_dim, hidden_dim).to(device)
        attn_decoder = AttentionDecoder(vocab_size, embedding_dim, hidden_dim, BATCH_NUM).to(device)

        encoder_model_path = './static//encoder_model.pth'
        att_decoder_model_path = './static/att_decoder_model.pth'

        encoder.load_state_dict(torch.load(encoder_model_path, map_location=torch.device('cpu')))
        attn_decoder.load_state_dict(torch.load(att_decoder_model_path, map_location=torch.device('cpu')))

        BATCH_NUM = 1

        # Decoderのアウトプットのテンソルから要素が最大のインデックスを返す。つまり生成文字を意味する
        def get_max_index(decoder_output):
            results = []
            for h in decoder_output:
                results.append(torch.argmax(h))
            return torch.tensor(results, device=device).view(BATCH_NUM, 1)

        input_tensor = torch.tensor(test_x, device=device).long()

        predicts = []
        with torch.no_grad():
            hs, encoder_state = encoder(input_tensor)

            # Decoderにはまず文字列生成開始を表す"_"をインプットにするので、"_"のtensorをバッチサイズ分作成
            start_char_batch = [[char2id["_"]] for _ in range(BATCH_NUM)]
            decoder_input_tensor = torch.tensor(start_char_batch, device=device)

            decoder_hidden = encoder_state

            batch_tmp = torch.zeros(1, 1, dtype=torch.long, device=device)
            for _ in range(50 - 1):
                decoder_output, decoder_hidden, _ = attn_decoder(decoder_input_tensor, hs, decoder_hidden)
                # 予測文字を取得しつつ、そのまま次のdecoderのインプットとなる
                decoder_input_tensor = get_max_index([decoder_output.squeeze()])
                batch_tmp = torch.cat([batch_tmp, decoder_input_tensor], dim=1)
            predicts.append(batch_tmp[:,1:])  

        # 予測結果を見る際にIDのままだと可読性が悪いので、もとの文字列に復元するためのID→文字列に変換する辞書を定義
        id2char = {}
        for k, v in char2id.items():
            id2char[v] = k

        for inp, predict in zip(test_x, predicts[0]):
            x = [id2char[idx] for idx in inp]
            p = [id2char[idx.item()] for idx in predict]

            x_str = "".join(x)
            p_str = "".join(p)

        result = {
                    'input':x_str,
                    'output':p_str
                }
        return JsonResponse(result)

class AIchatDetail(APIView):
    queryset = AIchat.objects.all()
    serializer_class = AIchatSerializer

    def get_object(self, pk):
        try:
            return AIchat.objects.get(pk=pk)
        except AIchat.DoesNotExist:
            raise Http404

    def get(self, request, pk, format=None):
        snippet = self.get_object(pk)
        serializer = AIchatSerializer(snippet)
        return Response(serializer.data)



@csrf_protect
def test(request):

    data = None
    try:
        data = json.loads(request.body)
    except ValueError:
        pass

    if 'text' in request.GET:
        TEXT_data = request.GET['text']
        result = {
                'input':TEXT_data,
                'output':'Success!'
            }

    elif data:
        result = {
                'input':data['text'],
                'output':'Success!'
            }
    
    else:
        result = {
                'input':'Nothing data!',
                'output':'Success!'
            }

    return JsonResponse(result)

