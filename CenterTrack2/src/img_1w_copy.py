import numpy as np
import os
import cv2
import ast
import sys
import shutil
DATA_PATH ='/home/ubuntu/cc/scripts/CenterTrack1/ann_data/test_img'
imge_path='/home/ubuntu/data/holo/holo/images'
images=os.listdir(imge_path)
images=sorted(images,key=lambda x:int(x.split('.')[0]))
VIDEO_SETS = {'train': range(8), 'test': range(3)}
path_txt=os.listdir(DATA_PATH)
path_txt=sorted(path_txt,key=lambda x:int(x.split('_')[0]))
for txt in path_txt:
    path1=os.path.join(DATA_PATH,txt)
    path4=path1.replace('ann_data','image')
    if not os.path.exists(path4):
        os.mkdir(path4)
    path2_list=os.listdir(path1)
    path2_list=sorted(list(map(lambda x:int(x.split('.')[0]),path2_list)))
    for res in path2_list:
        path7=os.path.join(imge_path,images[res])
        shutil.copy(path7,path4)
        