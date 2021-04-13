# TlabAPI-HD20
This API has been used  as Backend of the Hackday2020. This repository is for deploying to Azure server.

| URL          | Function           |
|:-------------|-------------------:|
| api/aichat/  | Call words from AI |
| api/ailearn/ | Tarin AI           |

##api/aichat/
###What is input? 
POST
'''JSON
{
  "text":"大島さん？"
}
'''
The vocabulary isn't perfect because of a little dataset, 
so input data can limit only japanese and don't use alphabet or a part of number.

###Output here
'''JSON
{
  "input":"大島さん？",
  "output":"児島だよ！"
}
'''

##api/ailearn/
###What is input?
'''form-data
{
  "epochtime":10
  "txtfile":boke_tukkomi.txt
}
'''
Upload data(.txt) must send from form-data, not json.

###Output here
{
    "vocabulary": 404,
    "input data": 77,
    "train data": 61.6,
    "epoch time": 10,
    "total loss": 995.9208223819733,
    "match ratio": 0.0
}
If this traing succeed, created these files att_decoder_model.pth, encoder_model.pth and char2id.csv under static file.
Train model data store in two .pth files and vocabulary data put in char2id.csv.
The dataset is divided into 80%:traindata and 20%:testdata.  

Warning!
Over 20 epoch time can occur timeout in Azure app server.
It is property that Azure server close over 230 seconds. 
Confirmed 30, 50, 100
