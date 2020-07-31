from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import sys
import cv2
import json
import copy
import numpy as np
from opts import opts
from detector import Detector

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge', 'display']

def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] ='0'
  opt.debug = max(opt.debug, 1)
  detector = Detector(opt)
  if opt.demo == 'webcam' or \
    opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
    is_video = True
    # demo on video stream
    cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
  else:
    is_video = False
    # Demo on images sequences
    if os.path.isdir(opt.demo):
      image_names = []
      ls = os.listdir(opt.demo)
      for file_name in sorted(ls,key=lambda x:int(x)):
        ll=os.path.join(opt.demo,file_name)
        ls1=os.listdir(ll)
        for file_name1 in sorted(ls1,key=lambda x:int(x.split('.')[0])):          
          ext = file_name1[file_name1.rfind('.') + 1:].lower()
          if ext in image_ext:
              image_names.append(os.path.join(ll, file_name1))
    else:
      image_names = [opt.demo]
  #image_names

  # Initialize output video
  out = None
  out_name = opt.demo.split('/')[-1]
  if opt.save_video:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('../results/{}.mp4'.format(
      opt.exp_id + '_' + out_name),fourcc, opt.save_framerate, (
        opt.video_w, opt.video_h))
  #opt.debug=1
  if opt.debug < 5:
    detector.pause = False
  cnt = 0
  results = {}

  while True:
      if is_video:
        _, img = cam.read()
        if img is None:
          save_and_exit(opt, out, results, out_name)
      else:
        if cnt < len(image_names):
          img = cv2.imread(image_names[cnt])
        else:
          save_and_exit(opt, out, results, out_name)
      cnt += 1

      # resize the original video for saving video results
      if opt.resize_video:
        img = cv2.resize(img, (opt.video_w, opt.video_h))

      # skip the first X frames of the video
      if cnt < opt.skip_first:
        continue
      
      #cv2.imshow('input', img)

      # track or detect the image.
      ret = detector.run(img)
      cv2.imshow('input', img)

      # log run time
      time_str = 'frame {} |'.format(cnt)
      #time_stats:['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge', 'display']
      for stat in time_stats:
        time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
      print(time_str)
      # results[cnt] is a list of dicts:
      #  [{'bbox': [x1, y1, x2, y2], 'tracking_id': id, 'category_id': c, ...}]
      #ret = {'results': [], 'tot': 0.0576939582824707,'load': 3.5762786865234375e-06, 'pre': 0.02759385108947754, 'net': 0.027010202407836914, 'dec': 0.002462625503540039, 'post': 0.0002605915069580078, 'merge': 7.295608520507812e-05, 'track': 0.0002849102020263672, 'display': 0.0221402645111084}
      results[cnt] = ret['results']

      # save debug image to video
      if opt.save_video:
        out.write(ret['generic'])
        if not is_video:
          cv2.imwrite('../results/demo{}.jpg'.format(cnt), ret['generic'])
      
      # esc to quit and finish saving video
      if cv2.waitKey(1) == 27:
        save_and_exit(opt, out, results, out_name)
        return 
  save_and_exit(opt, out, results)


def save_and_exit(opt, out=None, results=None, out_name=''):
  if opt.save_results and (results is not None):
    save_dir =  '../results/{}_results.json'.format(opt.exp_id + '_' + out_name)
    print('saving results to', save_dir)
    json.dump(_to_list(copy.deepcopy(results)), 
              open(save_dir, 'w'))
  if opt.save_video and out is not None:
    out.release()
  sys.exit(0)

def _to_list(results):
  for img_id in results:
    for t in range(len(results[img_id])):
      for k in results[img_id][t]:
        if isinstance(results[img_id][t][k], (np.ndarray, np.float32)):
          results[img_id][t][k] = results[img_id][t][k].tolist()
  return results

if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
