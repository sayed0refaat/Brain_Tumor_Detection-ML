import cv2
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
img_dir = 'dataset/'  # directory of dataset

no_tumor_images = os.listdir(img_dir+ 'no/') # open the dataset file then choose no file from it
yes_tumor_images = os.listdir(img_dir+ 'yes/') # open the dataset file then choose yes file from it
dataset=[]
label=[]

# print(no_tumor_images) #for example it prints no1.jpg

# path='no0.jpg'

# print(path.split('.')[1]) # here we extract what after the dot (jpg)

for i, img_name in enumerate(no_tumor_images):
    if( img_name.split('.')[1] == 'jpg' ):  # if we find extension jpg after the dot
        image = cv2.imread(img_dir+'no/'+img_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize((64,64))
        dataset.append(np.array(image))
        label.append(0)  # 0 stand for no

for i, img_name in enumerate(yes_tumor_images):
    if( img_name.split('.')[1] == 'jpg' ):  # if we find extension jpg after the dot
        image = cv2.imread(img_dir+'yes/'+img_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize((64,64))
        dataset.append(np.array(image))
        label.append(1)   # 1 stand for yes

#print(dataset)
#print(label)
#print(len(dataset))
#print(len(label))

dataset=np.array(dataset)
label=np.array(label)

x_train,x_test,y_train,y_test=train_test_split(dataset,label,test_size=0.2)    # test size = 20%

#print(x_train.shape)

