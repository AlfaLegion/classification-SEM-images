import torch
import os
import cv2
import torchvision
import numpy as np
import albumentations as aldu
import torchvision.transforms as transforms

class DataSet(torch.utils.data.Dataset):
    def __init__(self,dataRoot,transform=None):
       self.dataRoot=dataRoot
       self.transfrom=transform
       self.ToTensor=transforms.ToTensor()
       self.Normalize=transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
       self.classes=list()
       listClasses=os.listdir(self.dataRoot)
       self.dataset=list()

       for ndx in range(len(listClasses)):
           imgPath=os.path.join(self.dataRoot,listClasses[ndx])
           listImgs=os.listdir(imgPath)
           self.classes.append(listClasses[ndx])
           for imagename in listImgs:
               self.dataset.append({"input":os.path.join(imgPath,imagename),"label":ndx})



    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,index):

        input=cv2.imread(self.dataset[index]["input"],cv2.IMREAD_COLOR)
        input=input[0:725,:]
        tensor_label=torch.tensor(self.dataset[index]["label"])

        if self.transfrom is not None:
            input=self.transfrom(image=input)
            tensor_input=self.ToTensor(input['image'])
        else:
            tensor_input=self.ToTensor(input)
   
        tensor_input=self.Normalize(tensor_input)

        return tensor_input,tensor_label


def augDataset():
   batch_size=1
   pathSave="D:\\datasets\\DIBaS\\augDataset\\Enterococcus.faecium"
   if not os.path.exists(pathSave):
       os.mkdir(pathSave)
   transfrom=aldu.Compose([
        aldu.RandomCrop(256,256),
        #aldu.RandomContrast()
        ])
   dset=DataLoader("D:\\datasets\\DIBaS\\dataset\\qwe",transfrom)
   dloader=torch.utils.data.DataLoader(dataset=dset,
                                           batch_size=batch_size
                                           ,shuffle =False,
                                           num_workers=0)
   print(dset.list_classes)
   epoch=100
   count=0
   for cntr in range(epoch):
        for input in dloader:    
            a=input[0].numpy().copy()            
            count+=1
            nameImg=str(count)+".png"
            print(nameImg)
            cv2.imwrite(os.path.join(pathSave,nameImg),a)
            #cv2.imshow("image",a.astype(dtype=np.uint8))
            #cv2.waitKey()

 


if __name__=="__main__":

    augDataset()
    #batch_size=1

    #transfrom=aldu.Compose([
    #    aldu.RandomCrop(256,256),
    #    #aldu.RandomContrast()
    #    ])
    #dset=DataLoader("D:\\datasets\\DIBaS\\dataset\\qwe",transfrom)
    #dloader=torch.utils.data.DataLoader(dataset=dset,
    #                                       batch_size=batch_size
    #                                       ,shuffle =True,
    #                                       num_workers=0)
    #print(dset.list_classes)

    #for input,label in dloader:
    #    for ndx in range(batch_size):
    #        a=input[ndx].numpy().copy()
    #        a=a.transpose((2,1,0)).copy()
    #        print(label[ndx])
    #        a=a.transpose((1,0,2))
    #        a=cv2.normalize(a,None,0.,255.,cv2.NORM_MINMAX)
    #        cv2.imshow("image",a.astype(dtype=np.uint8))
    #        cv2.waitKey()




