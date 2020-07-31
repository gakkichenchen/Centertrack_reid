import os
import json
import cv2
import numpy as np
import sys
colours=np.random.randint(0,255,(255,3))
image='/home/ubuntu/cc/scripts/CenterTrack1/image/0_2019.04.250410_center_image_annotation_1'
ann_=open('/home/ubuntu/cc/scripts/CenterTrack1/ann_data/annotation_final/train.json','r')
img=os.listdir(image)
img=sorted(img,key=lambda x:int(x.split('.')[0]))
ann=json.load(ann_)
ann_ann=ann['annotations']
cv2.namedWindow('sh')
for i,im in enumerate(img):
    anns=ann_ann[i]
    frame=int(anns['image_id'])-1
    cv2.setWindowTitle("sh",img[frame])
    immmm=os.path.join(image,img[frame])
    resim=cv2.imread(immmm)
    track=anns['track_id']
    bbox=anns['bbox']
    color=list(colours[int(track)%255])
    cv2.rectangle(resim,(int(bbox[0]),int(bbox[1])),(int(bbox[2])+int(bbox[0]),int(bbox[3])+int(bbox[1])),(int(color[0]),int(color[1]),int(color[2])),2)
    cv2.imshow('sh',resim)
    if cv2.waitKey(30)==ord('q'):
        sys.exit(0)
    

