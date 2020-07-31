import json
import os
from collections import defaultdict
ann_=open('/home/ubuntu/cc/scripts/CenterTrack1/ann_data/annotations1/tracking_test.json','r')
path='/home/ubuntu/cc/scripts/CenterTrack1/ann_data/annotation_final'
ann=json.load(ann_)
ann_ann=ann['annotations']
result={-1:-1}
res=[]
num=defaultdict(list)
for anns in ann_ann:
    track=anns['track_id']
    if int(track) not in res:
        res.append(int(track))
res=sorted(res)
max_id=res[-1]
res=res[1:]
for i,ress in enumerate(res):
    result[ress]=i
print(result)
print(result[max_id])
path1=os.path.join(path,'test.json')
with open(path1,'w') as fn:
    num['images']=ann['images']
    for pa in ann_ann: 
        num['annotations'].append({'image_id':pa['image_id'],'id':pa['id'],'category_id':pa['category_id'],'bbox':pa['bbox'],'track_id':result[int(pa['track_id'])]})
    num['categories']=ann['categories']
    num['videos']=ann['videos']
    json.dump(num,fn)