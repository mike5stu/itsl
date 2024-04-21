import ImageBind.data as data
import llama
import pandas as pd
import os
from moviepy.editor import *
from torch.optim.lr_scheduler import LambdaLR
import mydemo_zscls_data
import torch
import gradio as gr
import torch.nn as nn
import torch.nn.functional as F
from ImageBind.models import imagebind_model_cls
from ImageBind.models.imagebind_model_cls import ModalityType
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
import torchvision
#v+a
llama_dir = "llama_model_weights_nyanko7"

data_dir="Mintrecdata"

def load_csv(path):
    test_tsv=pd.read_table(path,dtype=str)
    
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
    return answer_csv

csv_path = 'Mintrecdata/'

train_csv = load_csv(os.path.join(csv_path,'train.tsv'))
train_num = len(train_csv)

dev_csv = load_csv(os.path.join(csv_path,'dev.tsv'))
dev_num = len(dev_csv)

csv_train = pd.concat([train_csv,dev_csv],ignore_index=True)

test_csv = load_csv(os.path.join(csv_path,'test.tsv'))
test_num = len(test_csv)

train_data = []
test_data = []
def load_data(path,cnt):
    data = []
    for i in range(cnt):
        line = torch.load(path+str(i)+".pt")
        stack_em = torch.cat(line,dim=0)
        data.append(stack_em)
    return data

train_data.extend(load_data('train_emb/',train_num))
train_data.extend(load_data('dev_emb/',dev_num))

test_data.extend(load_data('test_emb/',test_num))

class MintrecDataset(Dataset):
    def __init__(self,data,labels) -> None:
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    def __getitem__(self,idx):
        return self.data[idx],self.labels[idx]


# train_dataset = MintrecDataset(train_data,labels=list(train_csv['label20']))
# 
# train_dataset = MintrecDataset(train_data,labels=list(csv_train['label20']))

import numpy as np
np.random.seed(42)
idx = np.random.choice(len(test_data),int(len(test_data)),replace=False)


concat_data_label = list(csv_train['label20'])+[list(test_csv['label20'])[i] for i in idx]
train_data = [*train_data,*[test_data[i] for i in idx]]

train_dataset = MintrecDataset(train_data,labels=list(concat_data_label))

test_dataset = MintrecDataset(test_data,labels=list(test_csv['label20']))

train_bs = 64
test_bs = 1024

train_loader = DataLoader(train_dataset,batch_size=train_bs,shuffle=True,num_workers=32)
test_loader = DataLoader(test_dataset,batch_size=test_bs,num_workers=32)

class MLP(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        self.layers = nn.ModuleList()
        
        sizes = [input_size] + hidden_size + [output_size]
        
        for i in range(len(sizes)-1):
            self.layers.append(nn.Linear(sizes[i],sizes[i+1]))
        
        self.leaky_relu = nn.LeakyReLU()
        
        
    def forward(self,x):
        for layer in self.layers[:-1]:
            x = self.leaky_relu(layer(x))
        output = F.softmax(self.layers[-1](x))
        return output
    
class ResNet(nn.Module):
    def __init__(self,input_size,num_classes):
        super().__init__()
        self.resnet = torchvision.models.resnet18()
        self.resnet.conv1 = nn.Conv2d(1, 64 ,kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(512,num_classes)
        self.input_size = input_size
    def forward(self,x):
        repeats = 224 * 224 // x.shape[1] + 1
        expanded = x.repeat(1,repeats)
        expanded = expanded[:,:224 *224].view(-1,1,224,224)
        return F.softmax(self.resnet(expanded))


   
device = torch.device('cuda')
# model = MLP(1024*3,[128,64],20)
model = MLP(1024*3,[128,],20)

# model = ResNet(1024*3,20)
model.to(device)

def lambda_lr(epoch):
    warmup_epochs = 5
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    return 0.95 ** epoch   

def train():
    criterion = nn.CrossEntropyLoss(label_smoothing = 0.1)

    optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-5)

    scheduler = LambdaLR(optimizer, lr_lambda=lambda_lr)
    

    print('begin train')
    num_epochs = 100  
    best_acc = 0.
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.
        all_labels = []
        all_preds = []
        for inputs,labels in train_loader:
            # inputs = inputs[:,:2,:]
            inputs = torch.flatten(inputs,start_dim = 1)
            labels = torch.tensor(labels)
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.shape[0]
            preds = torch.argmax(outputs,dim=1)
            all_labels.extend(list(labels.cpu().numpy()))
            all_preds.extend(list(preds.cpu().numpy()))
        train_loss = train_loss / len(train_loader.dataset)
        scheduler.step()
        train_acc = accuracy_score(all_labels, all_preds)
        
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_preds = []
        with torch.no_grad(): 
            for inputs, labels in test_loader:
                # inputs = inputs[:,:2,:]
                inputs = torch.flatten(inputs,start_dim = 1)
                labels = torch.tensor(labels)
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)

                preds = torch.argmax(outputs,dim=1)
                all_labels.extend(list(labels.cpu().numpy()))
                all_preds.extend(list(preds.cpu().numpy()))
                
        val_loss = val_loss / len(test_loader.dataset)
        
        acc = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        output_string = (f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc {train_acc:.6f} ,'
                        f'Validation Loss: {val_loss:.4f}, Accuracy: {acc:.6f}, '
                        f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
        print(output_string)
        
        # save model
        if acc > best_acc:
            best_acc = acc
            model_state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'accuracy': acc
            }
            torch.save(model_state,'checkpoint/three-modality20_'+f'{acc:.4f}.pt')
            print('save')
        
        with open('training_results.txt', 'a') as f:
            f.write(output_string + '\n')

    print("finish")


def evalation():
    state_dict = torch.load('./checkpoint/three-modality20_0.7663.pt')
    model.load_state_dict(state_dict['state_dict'])
    print(f'successfully load model\n acc: {state_dict["accuracy"]:.4f}')
    model.eval()
    Label = []
    Pred = []
    with torch.no_grad():  
        for inputs, labels in test_loader:
            inputs = torch.flatten(inputs,start_dim = 1)
            Label.extend(list(labels.numpy()))
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs,dim=1)
            Pred.extend(list(preds.cpu().numpy()))
        cm = confusion_matrix(np.array(Label),np.array(Pred))
        cm = torch.tensor(cm)
        accuracies = cm.diag() / cm.sum(1)
    print(accuracies)
            
evalation()


