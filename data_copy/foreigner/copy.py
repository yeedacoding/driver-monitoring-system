from data_copy.foreigner.data4yolo import *

# 개인 구글 드라이브 경로 지정
org_data_root = '/Volumes/G-DRIVE USB'
to_data_dir = 'dataset'

# split : train / valid / test
# version_num : 3, 4, 5, 6, 7
# data_type : images / labels
split_type = ['train', 'valid', 'test']
version_num = ['3', '4', '5', '6', '7']
data_type = ['images', 'labels']

for version in version_num :
    for data in data_type :
        checkFromDatasetCount(org_data_root, version, data)
        print('======================================================')
        for split in split_type :
            copyfile(org_data_root, to_data_dir, split, version, data)

copyYaml(org_data_root, to_data_dir)