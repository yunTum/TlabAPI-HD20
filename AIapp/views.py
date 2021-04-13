# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split
import random
from sklearn.utils import shuffle
import pandas as pd
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
from django.core.files.storage import FileSystemStorage
import os
#######Functions############################
def read_csv(file):
    data = {}
    with open(file,'r', encoding="utf_8") as f:
        csv_data = csv.reader(f)
        for row in csv_data:
            if not data:
                data = row
    return data

def write_csv(file, save_dict):
    save_row = {}
    with open(file,'w', encoding='utf_8') as f:
        writer = csv.DictWriter(f, fieldnames=save_dict.keys(),delimiter=",",quotechar='"')
        writer.writeheader()
        writer.writerows(save_row)


# Decoderのアウトプットのテンソルから要素が最大のインデックスを返す。つまり生成文字を意味する
def get_max_index(decoder_output, BATCH_NUM, device):
  results = []
  for h in decoder_output:
    results.append(torch.argmax(h))
  return torch.tensor(results, device=device).view(BATCH_NUM, 1)


# データをバッチ化するための関数を定義
def train2batch(input_data, output_data, batch_size=1):
    input_batch = []
    output_batch = []
    input_shuffle, output_shuffle = shuffle(input_data, output_data)
    for i in range(0, len(input_data), batch_size):
      input_batch.append(input_shuffle[i:i+batch_size])
      output_batch.append(output_shuffle[i:i+batch_size])
    return input_batch, output_batch

########################################


#####AI Classes###################################################################

# Encoderクラス
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, pad_idx):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, sequence):
        embedding = self.word_embeddings(sequence)
        # hsが各系列のGRUの隠れ層のベクトル
        # Attentionされる要素
        hs, h = self.gru(embedding)
        return hs, h

# Attention Decoderクラス
class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, batch_size, pad_idx):
        super(AttentionDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # hidden_dim*2としているのは、各系列のGRUの隠れ層とAttention層で計算したコンテキストベクトルをtorch.catでつなぎ合わせることで長さが２倍になるため
        self.hidden2linear = nn.Linear(hidden_dim * 2, vocab_size)
        # 列方向を確率変換したいのでdim=1
        self.softmax = nn.Softmax(dim=1)

    def forward(self, sequence, hs, h):
        embedding = self.word_embeddings(sequence)
        output, state = self.gru(embedding, h)
       # Attention層
       # hs.size() = ([100, 50, 128])
       # output.size() = ([100, 50, 128])

       # bmmを使ってEncoder側の出力(hs)とDecoder側の出力(output)をbatchごとまとめて行列計算するために、Decoder側のoutputをbatchを固定して転置行列を取る
        t_output = torch.transpose(output, 1, 2) # t_output.size() = ([100, 128, 10])

        # bmmでバッチも考慮してまとめて行列計算
        s = torch.bmm(hs, t_output) # s.size() = ([100, 50, 10])

        # 列方向(dim=1)でsoftmaxをとって確率表現に変換
        # この値を後のAttentionの可視化などにも使うため、returnで返しておく
        attention_weight = self.softmax(s) # attention_weight.size() = ([100, 50, 10])

        # コンテキストベクトルをまとめるために入れ物を用意
        c = torch.zeros(self.batch_size, 1, self.hidden_dim, device=self.device) # c.size() = ([100, 1, 128])

        # 各DecoderのGRU層に対するコンテキストベクトルをまとめて計算する方法がわからなかったので、
        # 各層（Decoder側のGRU層は生成文字列が10文字なので10個ある）におけるattention weightを取り出してforループ内でコンテキストベクトルを１つずつ作成する
        # バッチ方向はまとめて計算できたのでバッチはそのまま
        for i in range(attention_weight.size()[2]): # 10回ループ

          # attention_weight[:,:,i].size() = ([100, 50])
          # i番目のGRU層に対するattention weightを取り出すが、テンソルのサイズをhsと揃えるためにunsqueezeする
          unsq_weight = attention_weight[:,:,i].unsqueeze(2) # unsq_weight.size() = ([100, 50, 1])

          # hsの各ベクトルをattention weightで重み付けする
          weighted_hs = hs * unsq_weight # weighted_hs.size() = ([100, 50, 128])

          # attention weightで重み付けされた各hsのベクトルをすべて足し合わせてコンテキストベクトルを作成
          weight_sum = torch.sum(weighted_hs, axis=1).unsqueeze(1) # weight_sum.size() = ([100, 1, 128])

          c = torch.cat([c, weight_sum], dim=1) # c.size() = ([100, i, 128])

        # 箱として用意したzero要素が残っているのでスライスして削除
        c = c[:,1:,:]

        output = torch.cat([output, c], dim=2) # output.size() = ([100, 50, 256])
        output = self.hidden2linear(output)
        return output, state, attention_weight

##################################################################################



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

        encoder = Encoder(vocab_size, embedding_dim, hidden_dim, char2id["　"]).to(device)
        attn_decoder = AttentionDecoder(vocab_size, embedding_dim, hidden_dim, BATCH_NUM, char2id["　"]).to(device)

        encoder_model_path = './static//encoder_model.pth'
        att_decoder_model_path = './static/att_decoder_model.pth'

        encoder.load_state_dict(torch.load(encoder_model_path, map_location=torch.device('cpu')))
        attn_decoder.load_state_dict(torch.load(att_decoder_model_path, map_location=torch.device('cpu')))

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
                decoder_input_tensor = get_max_index([decoder_output.squeeze()], BATCH_NUM, device)
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

class AILearning(APIView):
#########################################
    queryset = AIchat.objects.all()
    serializer_class = AIchatSerializer
########################################### 
    def post(self, request, format=None):
        # boke_tukkomi.txt（訓練データ）をstatic下に保存
        lData = request.FILES['txtfile']
        fs = FileSystemStorage()
        fileurl = "static/" + lData.name
        # 上書きすると新たにリネームされたものが生成されるので、過去のデータを削除
        if fileurl:
            os.remove(fileurl)
        fs.save(fileurl, lData)
       # data
        file_path = "./static/boke_tukkomi.txt"

        input_date = [] # ボケデータ
        output_date = [] # ツッコミデータ

        # dataを1行ずつ読み込んでボケデータとツッコミに分割して、inputとoutputで分ける
        with open(file_path, "r",encoding="utf-8") as f:
            date_list = f.readlines()
            for date in date_list:
                date = date[:-1]
                input_date.append(date.split("_")[0])
                output_date.append("_" + date.split("_")[1])

        # inputとoutputの系列の長さを取得
        # すべて長さが同じなので、0番目の要素でlenを取ってます
        input_len = len(input_date[0]) # 29
        output_len = len(output_date[0]) # 10

        #規定ではない文字列のインデックスをprintで返す
        mistake_input = []
        mistake_output = []
        mistake_flag = False

        for i in range(len(input_date)):
            if len(input_date[i]) != 50:
                mistake_input.append(i+1)
                print("input: ",i+1)
                mistake_flag = True

        for i in range(len(output_date)):
            if len(output_date[i]) != 51:
                print("output: ",i+1)
                mistake_output.append(i+1)
                mistake_flag = True

        if mistake_flag:
            result = {
                "input: ":mistake_input,
                "output: ":mistake_output
            }
            return JsonResponse(result)

        # dataで登場するすべての文字にIDを割り当てる
        char2id = {}
        for input_chars, output_chars in zip(input_date, output_date):
            for c in input_chars:
                if not c in char2id:
                    char2id[c] = len(char2id)
            for c in output_chars:
                if not c in char2id:
                    char2id[c] = len(char2id)

        input_data = [] # ID化された変換前日付データ
        output_data = [] # ID化された変換後日付データ
        for input_chars, output_chars in zip(input_date, output_date):
            input_data.append([char2id[c] for c in input_chars])
            output_data.append([char2id[c] for c in output_chars])

        #文字にIDを振り分けたデータをcsvで保存
        write_csv('./static/char2id.csv', char2id)

        # 8:2でtrainとtestに分ける
        train_x, test_x, train_y, test_y = train_test_split(input_data, output_data, train_size= 0.8)

        # 諸々のパラメータなど
        embedding_dim = 200
        hidden_dim = 128
        BATCH_NUM = 1
        EPOCH_NUM = int(request.data['epochtime'])

        vocab_size = len(char2id)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        encoder = Encoder(vocab_size, embedding_dim, hidden_dim, char2id["　"]).to(device)
        attn_decoder = AttentionDecoder(vocab_size, embedding_dim, hidden_dim, BATCH_NUM, char2id["　"]).to(device)

        # 損失関数
        criterion = nn.CrossEntropyLoss()

        # 最適化
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
        attn_decoder_optimizer = optim.Adam(attn_decoder.parameters(), lr=0.001)

        all_losses = []
        print("training ...")

        for epoch in range(1, EPOCH_NUM+1):
            epoch_loss = 0
            # データをミニバッチに分ける
            input_batch, output_batch = train2batch(train_x, train_y, batch_size=BATCH_NUM)
            for i in range(len(input_batch)):

                # 勾配の初期化
                encoder_optimizer.zero_grad()
                attn_decoder_optimizer.zero_grad()

                # データをテンソルに変換
                input_tensor = torch.tensor(input_batch[i], device=device)
                output_tensor = torch.tensor(output_batch[i], device=device)

                # Encoderの順伝搬
                hs, h = encoder(input_tensor)

                # Attention Decoderのインプット
                source = output_tensor[:, :-1]

                # Attention Decoderの正解データ
                target = output_tensor[:, 1:]

                loss = 0
                decoder_output, _, attention_weight= attn_decoder(source, hs, h)
                for j in range(decoder_output.size()[1]):
                    loss += criterion(decoder_output[:, j, :], target[:, j])

                epoch_loss += loss.item()

                # 誤差逆伝播
                loss.backward()

                # パラメータ更新
                encoder_optimizer.step()
                attn_decoder_optimizer.step()

            # 損失を表示
            print("Epoch %d: %.2f" % (epoch, epoch_loss))
            all_losses.append(epoch_loss)
            if epoch_loss < 0.1: break
        print("Done")

        # 評価用データ
        test_input_batch, test_output_batch = train2batch(test_x, test_y)
        input_tensor = torch.tensor(test_input_batch, device=device)

        predicts = []
        for i in range(len(test_input_batch)):
            with torch.no_grad():
                hs, encoder_state = encoder(input_tensor[i])

                # Decoderにはまず文字列生成開始を表す"_"をインプットにするので、"_"のtensorをバッチサイズ分作成
                start_char_batch = [[char2id["_"]] for _ in range(BATCH_NUM)]
                decoder_input_tensor = torch.tensor(start_char_batch, device=device)

                decoder_hidden = encoder_state

                batch_tmp = torch.zeros(1, 1, dtype=torch.long, device=device)

                for _ in range(output_len - 1):
                    decoder_output, decoder_hidden, _ = attn_decoder(decoder_input_tensor, hs, decoder_hidden)
                    # 予測文字を取得しつつ、そのまま次のdecoderのインプットとなる
                    decoder_input_tensor = get_max_index([decoder_output.squeeze()], BATCH_NUM, device)
                    
                    batch_tmp = torch.cat([batch_tmp, decoder_input_tensor], dim=1)
            predicts.append(batch_tmp[:,1:])  

        # 予測結果を見る際にIDのままだと可読性が悪いので、もとの文字列に復元するためのID→文字列に変換する辞書を定義
        id2char = {}
        for k, v in char2id.items():
            id2char[v] = k

        row = []
        for i in range(len(test_input_batch)):
            batch_input = test_input_batch[i]
            batch_output = test_output_batch[i]
            batch_predict = predicts[i]
            for inp, output, predict in zip(batch_input, batch_output, batch_predict):
                x = [id2char[idx] for idx in inp]
                y = [id2char[idx] for idx in output[1:]]
                p = [id2char[idx.item()] for idx in predict]

                x_str = "".join(x)
                y_str = "".join(y)
                p_str = "".join(p)

                judge = "O" if y_str == p_str else "X"
                row.append([x_str, y_str, p_str, judge])
        predict_df = pd.DataFrame(row, columns=["input", "answer", "predict", "judge"])
        predict_df.to_csv("./static/predict.csv", encoding="shift_jis")

        encoder_model_path = './static/encoder_model.pth'
        att_decoder_model_path = './static/att_decoder_model.pth'
        torch.save(encoder.to('cpu').state_dict(), encoder_model_path)
        torch.save(attn_decoder.to('cpu').state_dict(), att_decoder_model_path)

        result = {
                    'match':len(predict_df.query('judge == "O"')) / len(predict_df),
                }
        return JsonResponse(result)


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

