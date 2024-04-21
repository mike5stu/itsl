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
test_tsv=pd.read_table("Mintrecdata/dev.tsv",dtype=str)
test_tsv["file_path"]=data_dir+"/raw_data/"+test_tsv["season"]+"/"+test_tsv["episode"]+"/"+test_tsv["clip"]+".mp4"

# print(test_tsv)
# print(test_tsv["file_path"])

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
answer_csv["output"]={}
answer_csv["result"]={}
answer_csv["out_class"]={}
for iter in range(len(test_tsv)):
    # print(iter, test_tsv["label"][iter],end=" ")
    if test_tsv["label"][iter] in ["Complain", "Praise", "Apologise", "Thank", "Criticize", "Care", "Agree", "Taunt", "Flaunt", "Oppose", "Joke"]:
        # print("express")
        answer_csv["label2"][iter]="express"
        answer_csv["label20"][iter]=test_tsv["label"][iter]
    elif test_tsv["label"][iter] in ["Inform", "Advise", "Arrange", "Introduce", "Comfort", "Leave", "Prevent", "Greet", "Ask for help"]:
        answer_csv["label2"][iter]="achieve"
        answer_csv["label20"][iter]=test_tsv["label"][iter]
        # print("achieve")
# print(answer_csv)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = imagebind_model_cls.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

def video_text_zeroshot(video,audio, text_list):
    video_paths = [video]
    audio_paths = [audio]
    labels = [text_list]
    inputs = {
        ModalityType.TEXT: mydemo_zscls_data.load_and_transform_text(labels, device),
        ModalityType.VISION: mydemo_zscls_data.load_and_transform_video_data(video_paths, device),
        ModalityType.AUDIO: mydemo_zscls_data.load_and_transform_audio_data(audio_paths, device),
    }
    with torch.no_grad():
        embeddings = model(inputs)
    return [embeddings[ModalityType.TEXT].cpu(),embeddings[ModalityType.VISION].cpu(),embeddings[ModalityType.AUDIO].cpu()]

# print(video_text_zeroshot(test_tsv["file_path"][0],"complain|praise|apologise|thank|criticize|care|agree|taunt|flaunt|oppose|joke|inform|advise|arrange|introduce|comfort|leave|prevent|greet|ask for help"))
# print(video_text_zeroshot(test_tsv["file_path"][0],"Express emotions or attitudes|Achieve goals"))

# text_list="complain|praise|apologise|thank|criticize|care|agree|taunt|flaunt|oppose|joke|inform|advise|arrange|introduce|comfort|leave|prevent|greet|ask for help"

for i in range(len(test_tsv)):#len(test_tsv)
    samp_num=i
    print(test_tsv["text"][samp_num],test_tsv["label"][samp_num])
    # print(test_tsv["text"][samp_num]+" Express emotions or attitudes|"+test_tsv["text"][samp_num]+" Achieve goals")
    result=video_text_zeroshot(test_tsv["file_path"][samp_num], test_tsv["file_path"][samp_num] + ".wav",test_tsv["text"][samp_num])
    print(result)
    torch.save(result,"dev_emb/"+str(samp_num)+".pt")
    # answer_csv["output_V"][i]=result[0]
    # answer_csv["out_class"][i]=process_txt_modal(result_txt)
    # print(result)

# print(answer_csv)


# answer_csv.to_csv("test_emb.csv")
