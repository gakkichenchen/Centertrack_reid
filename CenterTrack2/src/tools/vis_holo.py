import numpy as np
import cv2
import os
import glob
import sys
from collections import defaultdict
from pathlib import Path
import sys
import argparse

DATA_PATH ='/home/ubuntu/cc/scripts/CenterTrack1/result_true'
IMG_PATH ='/home/ubuntu/cc/scripts/CenterTrack1/image/test_img'
SAVE_VIDEO = False
IS_GT = False

cats = ['car', 'people']
cat_ids = {cat: i for i, cat in enumerate(cats)}
COLORS =np.random.randint(0,255,(255,3))
def parse_arg():
    parse=argparse.ArgumentParser(description='SORT')
    parse.add_argument('--num',type=int)
    args=parse.parse_args()
    return args
def draw_bbox(img, bboxes, c=(255, 0, 255)):
  for bbox in bboxes:
    color = COLORS[int(bbox[4])%255]
    color1=(int(color[0]),int(color[1]),int(color[2]))
    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), 
      (int(bbox[2]), int(bbox[3])), color1, 2, lineType=cv2.LINE_AA)
    ct = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    txt = '{}'.format(int(bbox[4]))
    cv2.putText(img, txt, (int(ct[0]), int(ct[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color1, thickness=1, lineType=cv2.LINE_AA)

if __name__ == '__main__':
  p=parse_arg()
  num=p.num
  f=os.listdir(IMG_PATH)
  seqs = sorted(f,key=lambda x:int(x.split('_')[0]))
  path={}
  for i,seq in enumerate(seqs):
    path[i]=defaultdict(list)
    txt=seq+'.txt'
    txtpath=os.path.join(DATA_PATH,txt)
    pretxt=open(txtpath,'r')
    for line in pretxt:
      tmp = line[:-1].split(' ')
      frame_id = int(tmp[0])
      track_id = int(tmp[1])
      cat_id = cat_ids[tmp[2]]
      bbox = [float(tmp[6]), float(tmp[7]), float(tmp[8]), float(tmp[9])]
      score = float(tmp[17])
      path[i][frame_id].append(bbox + [track_id, cat_id, score])
  cv2.namedWindow('img')
  num1=len(seqs)
  if num:
    num1=1
  for s in range (num1):
    if num:
      s=num
    seq=seqs[s]
    imgpath1=os.path.join(IMG_PATH,seq)
    im=os.listdir(imgpath1)
    im=sorted(im,key=lambda x:int(x.split('.')[0]))
    for j,imm in enumerate(im):
      cv2.setWindowTitle('img',imm)
      imgpath=os.path.join(imgpath1,imm)
      img=cv2.imread(imgpath)
      draw_bbox(img,path[s][j])
      cv2.imshow('img',img)
      if cv2.waitKey(30)==ord('q'):
        sys.exit(0)
        