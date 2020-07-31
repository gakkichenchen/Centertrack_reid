import os
import shutil
ann_path='/home/ubuntu/cc/scripts/CenterTrack1/20200713/ann/01'
video_path='/home/ubuntu/cc/scripts/CenterTrack/imgfile/test_img'
video='/home/ubuntu/cc/scripts/CenterTrack1/ann_data'
list_=os.listdir(video_path)
ann_list=os.listdir(ann_path)
for lists in list_:
    path2=os.path.join(video_path,lists)
    img_list=os.listdir(path2)
    img_list=list(map(lambda x:x.replace('jpeg','txt').replace('jpg','txt').replace('png','txt'),img_list))
    for img in img_list:
        path3=os.path.join(ann_path,img)
        if os.path.exists(path3):
            path1=os.path.join(video,lists)
            if not os.path.exists(path1):
                os.mkdir(path1)
            shutil.copy(path3,path1)
    
    