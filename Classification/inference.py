
import torch
import numpy as np
import torchvision.transforms as transforms
import DataLoader.DataLoader as DataLoader
import os
import cv2
import torch.nn.functional as F


def validate (model, dataloader):
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


def inference_folder(rootData,pathModel="clsf.pt",map_dev="cuda"):

    save_error_path=r'D:/datasets/11_06_2019_wtn/dataset2\error'

    device = torch.device(map_dev)
    model = torch.jit.load(pathModel,map_dev)

    tranform=transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])


    listClass=os.listdir(rootData)
    print(listClass)

    for current_class in listClass:
        path_class=os.path.join(rootData,current_class)
        list_content=os.listdir(path_class)
        error_file=os.path.join(save_error_path,current_class)

        if not os.path.exists(error_file):
            os.makedirs(error_file)
        
        for data_name in list_content:
            current_sample=os.path.join(path_class,data_name)
            image=cv2.imread(current_sample,cv2.IMREAD_COLOR)
            original=image.copy()
            image=image[0:725,:]

            tensor_input=tranform(image).unsqueeze_(0).cuda()
            output=model(tensor_input)
            predict=F.softmax(output,dim=1)
            predict=predict.cpu().detach().numpy()
            result=np.argmax(predict)

            print(listClass[result], "\n",predict)
            if(listClass[result]!=current_class):
                error_image=os.path.join(error_file,listClass[result]+'.tif')
                cv2.imwrite(error_image,original)

    #image_datasets_test=DataLoader.DataSet(rootData,transform=None)
    #print(image_datasets_test.class_name())
    #dataLoaderTest=torch.utils.data.DataLoader(image_datasets_test, batch_size=int(5),shuffle=False, num_workers=0)
    #validate(model,dataLoaderTest)

if __name__=='__main__':
    inference_folder(r'D:/datasets/11_06_2019_wtn/dataset2/test','ClassifyModelResNet34.pt')

