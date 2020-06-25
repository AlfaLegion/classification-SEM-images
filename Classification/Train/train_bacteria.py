import torch
import os
import cv2
import torch.nn as nn
import torchvision.models as models
import numpy as np
import albumentations as albu
import DataLoader.DataLoader as DataLoader
from torch.optim import lr_scheduler
import torch.optim as optim
import visdom
import copy
from torchsummary import summary
def convertModel(model,modelName,img_sz):
    print("Start...")
    device = torch.device('cuda')
    #model = torch.load(SavePathModel,map_location=device)
    #model = model.float()
    #model.eval()
    example = torch.rand(1, 3, img_sz[0],img_sz[1],device=device)
    model.eval()
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save(modelName)
    print("Complete")


def test (model, dataloader):
    with torch.no_grad():
        model.eval()
        acc=0.0
        num=0
        cnt_smpls=0
        cnt_test_samples=len(dataloader)
        for input, label in dataloader:
            input=input.cuda()
            label=label.cuda()
            outputs=model(input)
            _, predict = torch.max(outputs, 1)
            acc+=torch.sum(predict==label.data).item()
            cnt_smpls+=input.shape[0]
            num+=1
        model.train()
        return acc/cnt_smpls

def get_model_resNet_34():
    model=models.resnet34(True)
    model.fc=nn.Linear(512,7,True)
    return model

def get_model_resNet_18():
    model=models.resnet18(True)
    model.fc=nn.Linear(512,7,True)
    return model

def get_model_mobile_net_v2():
    model=models.mobilenet_v2(True)
    model.classifier[1]=nn.Linear(1280,7,True)
    return model

def train(dataDir,testDir,nameModel,num_epoch,batch_size,lr,momentum,weight_decay):
    print("dataDir: ",dataDir)
    print("num_epoch: ",num_epoch)
    print("batch_size: ",batch_size)
    print("learning rate: ",lr)
    print("momentum: ",momentum)

    data_transform =albu.Compose([
            #albu.Resize(362,512),
            albu.Rotate(),
            albu.HorizontalFlip(),
            albu.VerticalFlip(),
            albu.RandomContrast(),
            albu.RandomGamma(),
            ])
    image_datasets = DataLoader.DataSet(dataDir,transform=data_transform)
    image_datasets_test=DataLoader.DataSet(testDir,transform=None)

    ds_size_train=len(image_datasets)
    ds_size_test=len(image_datasets_test)
    print("size train dataset: ",ds_size_train)
    print("size test dataset: ",ds_size_test)

    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=int(batch_size),shuffle=True, num_workers=6)
    dataLoaderTest=torch.utils.data.DataLoader(image_datasets_test, batch_size=int(5),shuffle=False, num_workers=6)

    class_names = image_datasets.classes
    print(class_names)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Select cuda device")
    else:
        print("Select cpu")
        
    criterion = nn.CrossEntropyLoss()
    #model=get_model()
    #model=get_model_resNet_18()
    #model=get_model_resNet_34()
    model=get_model_mobile_net_v2()
    model = model.to(device)
    model.train()
    
    optimizer = optim.SGD(model.parameters(), lr=float(lr), momentum=float(momentum),weight_decay=weight_decay)
    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    epoch=int(num_epoch)

    vis=visdom.Visdom()
    averageLoss=vis.line(np.array([0]),np.array([0]),opts=dict(xlabel="epoch",ylable="loss",title="Average loss per epoch"))
    Test=vis.line(np.array([0]),np.array([0]),opts=dict(xlabel="epoch",ylable="Acc",title="Acc test"))

    best_acc=0.0
    for ep in range(epoch):
        running_loss=0
        print("epoch: ",ep+1,"/",epoch)
        for  input, label in dataloaders:

            optimizer.zero_grad()
            input = input.to(device)
            label = label.to(device)

            output = model(input)
            loss = criterion(output,label)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
        acc=test(model,dataLoaderTest)
        vis.line(np.array([running_loss/ds_size_train]),np.array([ep]),win=averageLoss,update="append")
        vis.line(np.array([acc]),np.array([ep]),win=Test,update="append")
        #exp_lr_scheduler.step()
        print(running_loss/ds_size_train)
        if best_acc<acc:
            best_acc=acc
            best_model = copy.deepcopy(model.state_dict())

    print("best score: ", best_acc)
    #convertModel(best_model,nameModel,(256,256))



import time as t



if __name__=="__main__":
       
    #train(r'D:/datasets/11_06_2019_wtn/dataset2/train',
    #      r'D:/datasets/11_06_2019_wtn/dataset2/test',
    #     'ClassifyModelResNet18',
    #      100,7,0.0001,0.9,0.0005)

