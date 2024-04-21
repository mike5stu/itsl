import ImageBind.data as data
import llama
import pandas as pd
import os
from moviepy.editor import *
import mydemo_zscls_data
import torch
import gradio as gr
from ImageBind.models import imagebind_model_cls
from ImageBind.models.imagebind_model_cls import ModalityType
#v+a
llama_dir = "llama_model_weights_nyanko7"

data_dir="Mintrecdata"

test_tsv=pd.read_table("train.tsv",dtype=str)
test_tsv["file_path"]=data_dir+"/raw_data/"+test_tsv["season"]+"/"+test_tsv["episode"]+"/"+test_tsv["clip"]+".mp4"

for iter in test_tsv["file_path"]:
    if not os.path.isfile(iter):
        print(iter+" not exists")
        exit(0)
    if not os.path.isfile(iter+".wav"):
        input_file = iter
        output_file = iter+".wav"
        sound = AudioFileClip(input_file)
        sound.write_audiofile(output_file, 44100, 2, 2000,"pcm_s32le")

print("all file exists")
answer_csv=pd.DataFrame()
answer_csv["label2"]=[x for x in range(len(test_tsv))]
answer_csv["label20"]={}
answer_csv["result"]={}

for iter in range(len(test_tsv)):
    # print(iter, test_tsv["label"][iter],end=" ")
    if test_tsv["label"][iter] in ["Complain", "Praise", "Apologise", "Thank", "Criticize", "Care", "Agree", "Taunt", "Flaunt", "Oppose", "Joke"]:
        # print("express")
        answer_csv["label2"][iter]=0
        answer_csv["label20"][iter]=test_tsv["label"][iter]
    elif test_tsv["label"][iter] in ["Inform", "Advise", "Arrange", "Introduce", "Comfort", "Leave", "Prevent", "Greet", "Ask for help"]:
        answer_csv["label2"][iter]=1
        answer_csv["label20"][iter]=test_tsv["label"][iter]
        # print("achieve")
answer_csv["label20"] = answer_csv["label20"].map({"Complain":0,"Praise":1,"Apologise":2,"Thank":3,"Criticize":4,"Care":5,"Agree":6,"Taunt":7,"Flaunt":8,"Oppose":9,"Joke":10,"Inform":11,"Advise":12,"Arrange":13,"Introduce":14,"Comfort":15,"Leave":16,"Prevent":17,"Greet":18,"Ask for help":19})



for i in range(10):#len(test_tsv)
    samp_num=i
    print(test_tsv["text"][samp_num],test_tsv["label"][samp_num])

    print(torch.load("dev_emb/"+str(samp_num)+".pt"))


print(answer_csv)


# answer_csv.to_csv("test_emb.csv")
# print(answer_csv)
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# model = imagebind_model_cls.imagebind_huge(pretrained=True)
# model.eval()
# model.to(device)

# print(video_text_zeroshot(test_tsv["file_path"][0],"complain|praise|apologise|thank|criticize|care|agree|taunt|flaunt|oppose|joke|inform|advise|arrange|introduce|comfort|leave|prevent|greet|ask for help"))
# print(video_text_zeroshot(test_tsv["file_path"][0],"Express emotions or attitudes|Achieve goals"))

# text_list="complain|praise|apologise|thank|criticize|care|agree|taunt|flaunt|oppose|joke|inform|advise|arrange|introduce|comfort|leave|prevent|greet|ask for help"
    # answer_csv["output_V"][i]=result[0]
    # answer_csv["out_class"][i]=process_txt_modal(result_txt)
    # print(result)
    # print(test_tsv["text"][samp_num]+" Express emotions or attitudes|"+test_tsv["text"][samp_num]+" Achieve goals")
    # result=video_text_zeroshot(test_tsv["file_path"][samp_num], test_tsv["file_path"][samp_num] + ".wav",test_tsv["text"][samp_num])
    # print(result)