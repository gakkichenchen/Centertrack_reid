import json
import os
from collections import defaultdict
list_=['/home/ubuntu/cc/scripts/CenterTrack/imgfile/train_img/76_201910080812_center_ring_annotation_3_2/78179.jpeg', '/home/ubuntu/cc/scripts/CenterTrack/imgfile/train_img/64_2019.02.280226_rear_image_annotation_1/69532.jpeg', '/home/ubuntu/cc/scripts/CenterTrack/imgfile/train_img/36_2019.01.240124_right_image_annotation_04_2/39801.jpeg', '/home/ubuntu/cc/scripts/CenterTrack/imgfile/train_img/78_201910080812_center_ring_annotation_7/80065.jpeg', '/home/ubuntu/cc/scripts/CenterTrack/imgfile/train_img/46_2019.02.200215_side_image_annotation_01_2/50557.jpg', '/home/ubuntu/cc/scripts/CenterTrack/imgfile/train_img/29_2019.01.240124_center_image_annotation_07_2/33313.jpeg', '/home/ubuntu/cc/scripts/CenterTrack/imgfile/train_img/64_2019.02.280226_rear_image_annotation_1/69791.jpeg', '/home/ubuntu/cc/scripts/CenterTrack/imgfile/train_img/4_2019.04.250410_right_image_annotation_1/6150.jpeg', '/home/ubuntu/cc/scripts/CenterTrack/imgfile/train_img/5_2019.04.250410_right_image_annotation_2/8197.jpeg', '/home/ubuntu/cc/scripts/CenterTrack/imgfile/train_img/64_2019.02.280226_rear_image_annotation_1/69704.jpeg', '/home/ubuntu/cc/scripts/CenterTrack/imgfile/train_img/32_2019.01.240124_left_image_annotation_05_2/35910.jpg', '/home/ubuntu/cc/scripts/CenterTrack/imgfile/train_img/0_2019.04.250410_center_image_annotation_1/507.jpeg', '/home/ubuntu/cc/scripts/CenterTrack/imgfile/train_img/10_train1123_center_image_annotation_2/14497.jpeg', '/home/ubuntu/cc/scripts/CenterTrack/imgfile/train_img/67_201909100812_center_ring_annotation_2_1/72068.jpeg', '/home/ubuntu/cc/scripts/CenterTrack/imgfile/train_img/41_2019.02.200215_left_image_annotation_08_2/45709.jpeg', '/home/ubuntu/cc/scripts/CenterTrack/imgfile/train_img/64_2019.02.280226_rear_image_annotation_1/69614.jpeg']
ann_path='/home/ubuntu/cc/scripts/CenterTrack/ann/tracking_train.json'
result=defaultdict(list)
f=open(ann_path)
json_data=json.load(f)
with open('/home/ubuntu/cc/scripts/CenterTrack1/ex_img/ex.json','w') as fn:
    for lists in list_:
        img1=lists.split('/')[-1]
        img2=lists.split('/')[-2]
        path3=os.path.join(img2,img1)
        num=int(img1.split(".")[0])
        for res in json_data['images']:
            if res['file_name']==path3:
                result['images'].append(res)
        for res in json_data['annotations']:
            if res['image_id']==num+1:
                result['annotations'].append(res)
        result['categories']=json_data['categories']
        for res in json_data['videos']:
            if res['file_name']==img2:
                result['videos'].append(res)
    json.dump(result,fn)
fn.close()
fmmm=open('/home/ubuntu/cc/scripts/CenterTrack1/ex_img/ex.json')
fmmm=json.load(fmmm)
import ipdb; ipdb.set_trace()
print(fmmm)
