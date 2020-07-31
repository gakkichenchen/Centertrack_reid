import os
import json
DATA_PATH = '/home/ubuntu/cc/scripts/sort-master/outputjson/test'
json_list=os.listdir(DATA_PATH)
json_list=sorted(json_list,key=lambda x:int(x.split('_')[0]))
for i in range(len(json_list)):
    json_path=os.path.join(DATA_PATH,json_list[i])
    f=open(json_path)
    files=json.load(f)
    with open ('/home/ubuntu/cc/scripts/sort-master/outputjson/test_txt/{}.txt'.format(json_list[i].split('.')[0]),'w') as resf:
        frame=-1
        ress=-1
        for j,file1 in enumerate(files):
            frame+=1
            if not file1['label']:
                continue
            for m,res in enumerate(file1['label']):
                ress+=1
                if ress==0 and m==0:
                    id1=res['id']
                x0=res['box2d']['x0']
                y0=res['box2d']['y0']
                x1=res['box2d']['x1']
                y1=res['box2d']['y1']
                if 'car' in res['categary']:
                    res['categary']='car'
                resf.write('{} {} {} -1 -1 -1 {} {} {} {} -1 -1 -1 -1000 -1000 -1000 -10 1\n'.format(frame,res['id']-id1+1,res['categary'],x0,y0,x1,y1))
    resf.close()   
