from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import json
import numpy as np
import os
import cv2
import ast
import sys
DATA_PATH ='/home/ubuntu/cc/scripts/CenterTrack1/ann_data'
SPLITS = ['train','test']
VIDEO_SETS = {'train': range(8), 'test': range(3)}
CREATE_HALF_LABEL = True
DEBUG = False
def _bbox_to_coco_bbox(bbox):
  return [(bbox[0]), (bbox[1]),(bbox[2] - bbox[0]), (bbox[3] - bbox[1])]
cat_info=[]
cats={'car':1,'people':2,'others':3}
for i, cat in enumerate(cats):
  cat_info.append({'name': cat, 'id': i + 1})
if __name__ == '__main__':
  maxnum=0
  for split in SPLITS:
    video_list=os.listdir(DATA_PATH+'/{}_img'.format(split))
    video_list=sorted(video_list,key=lambda x:int(x.split('_')[0]))
    ret = {'images': [], 'annotations': [], "categories": cat_info,
           'videos': []}
    num_images = 0
    for i in VIDEO_SETS[split]:
      video_name = video_list[i]
      ret['videos'].append({'id': i + 1, 'file_name': video_name})
      ann_dir = 'train' if not ('test' in split) else split
      video_path = DATA_PATH + '/{}_img/{}'.format(ann_dir, video_name)
      ann_=sorted(os.listdir(video_path),key=lambda x:int(x.split('.')[0]))
      video_path1=video_path.replace('ann_data','image')
      image_files = sorted(os.listdir(video_path1),key=lambda x:int(x.split('.')[0]))
      for j, image_name in enumerate(ann_):
        num_images += 1
        image_info = {'file_name': '{}/{}'.format(video_name,image_files[j]),
                      'id': num_images,
                      'video_id': i + 1,
                      'frame_id': j + 1 }
        ret['images'].append(image_info)
        img=image_name.split('.')[0]
        ann_path = DATA_PATH + '/{}_img/{}/{}.txt'.format(split,video_name,img)
        f=open(ann_path)
        for fn in f:
          if fn=='[]':
            ret['annotations'].append({'image_id':num_images,'id':int(len(ret['annotations'])+1),'category_id':3,'bbox':[0.4,0.3,0.2,0.1],'track_id':-1})
          list_=ast.literal_eval(fn)
          for ann_ind in list_:
            if ann_ind['category']=='37' or ann_ind['category']=='36':
              cat=1
            elif ann_ind['category']=='25':
              cat=2
            else:
              cat=-1   
            if int(ann_ind['id'])>maxnum:
              maxnum=int(ann_ind['id'])
            ann = {'image_id': num_images ,
                        'id': int(len(ret['annotations']) + 1),
                        'category_id': cat,
                        'bbox': _bbox_to_coco_bbox(ann_ind['points']),
                        'track_id':ann_ind['id']}
            ret['annotations'].append(ann)
    
    out_dir = '{}/annotations1/'.format(DATA_PATH)
    if not os.path.exists(out_dir):
      os.mkdir(out_dir)
    out_path = '{}/annotations1/tracking_{}.json'.format(DATA_PATH, split)
    json.dump(ret, open(out_path, 'w'))
    print(maxnum)
