import os
img_path_train='/home/ubuntu/cc/scripts/sort-master/output/train_img'
train_name=os.listdir(img_path_train)
train_name=sorted(train_name,key=lambda x:int(x.split('_')[0]))

img_path_train1='/home/ubuntu/cc/scripts/sort-master/output/test_img'
train_name1=os.listdir(img_path_train1)
train_name1=sorted(train_name1,key=lambda x:int(x.split('_')[0]))

with open('../holo_tracking/holo_data_train.txt','w') as f:
    for name in train_name:
        f.write('{} {}\n'.format(name,len(os.listdir(os.path.join(img_path_train,name)))))
with open('../holo_tracking/holo_data_test.txt','w') as f:
    for name in train_name1:
        f.write('{} {}\n'.format(name,len(os.listdir(os.path.join(img_path_train1,name)))))