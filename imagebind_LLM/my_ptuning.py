import ImageBind.data as data
import llama
import pandas as pd
import os
from moviepy.editor import *

llama_dir = ""

data_dir="Mintrecdata"
train_tsv=pd.read_table("train.tsv",dtype=str)
train_tsv["file_path"]=data_dir+"/raw_data/"+train_tsv["season"]+"/"+train_tsv["episode"]+"/"+train_tsv["clip"]+".mp4"

# print(train_tsv)
# print(train_tsv["file_path"])

for iter in train_tsv["file_path"]:
    if not os.path.isfile(iter):
        print(iter+" not exists")
        exit(0)
    if not os.path.isfile(iter+".wav"):
        input_file = iter
        output_file = iter+".wav"
        sound = AudioFileClip(input_file)
        sound.write_audiofile(output_file, 44100, 2, 2000,"pcm_s32le")

print("all file exists")

