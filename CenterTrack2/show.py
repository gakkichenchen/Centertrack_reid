import os
import cv2
import numpy as np
import ast
import sys
path=('/home/ubuntu/cc/scripts/CenterTrack1/20200713/ann/01')
colours=np.random.randint(0,255,(255,3))
list_=os.listdir(path)
list_=sorted(list_,key=lambda x:int(x.split(".")[0]))
path_img='/home/ubuntu/data/holo/holo/images'
img=os.listdir(path_img)
img1=sorted(img,key=lambda x:int(x.split(".")[0]))
cv2.namedWindow("sh")
for num,lists in enumerate(list_):
    nums=int(lists.split(".")[0])
    path1=os.path.join(path,lists)
    with open(path1,'r') as file_:
        cv2.setWindowTitle("sh",img1[nums])
        fa=file_.readline()
        fa=ast.literal_eval(fa)
        if not fa:
            image=os.path.join(path_img,img1[nums])
            im=cv2.imread(image)
            cv2.imshow('sh',im)
            cv2.waitKey(30)
            continue
        image=os.path.join(path_img,img1[nums])
        im=cv2.imread(image)
        for line in fa:
            bbox=line['points']
            cat=line['id']
            CATA=line['category']
            if CATA!='36' and CATA!='37':
                print(CATA)
            color=list(colours[int(cat)%255])
            cv2.rectangle(im,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(int(color[0]),int(color[1]),int(color[2])),2)
        cv2.imshow('sh',im)
        if cv2.waitKey(30)==ord('q'):
            cv2.destroyWindow(im)
            sys.exit(0)
            
            
            
            
