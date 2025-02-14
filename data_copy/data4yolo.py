import os
import shutil
import glob
import yaml
from tqdm import tqdm


# 다운 받은 데이터셋들이 들어있는 경로에서 데이터 수 체크
# version_num : 3 / 4 / 5 / 6 / 7
# data_type : images / labels
def checkFromDatasetCount(root_dir, version_num, data_type) :
    from_data_dir = f'DMD.v{version_num}i.yolov11'

    print("train : ", len(os.listdir(f"{root_dir}/{from_data_dir}/train/{data_type}")))
    print("valid : ", len(os.listdir(f"{root_dir}/{from_data_dir}/valid/{data_type}")))
    print("test : ", len(os.listdir(f"{root_dir}/{from_data_dir}/test/{data_type}")))

    print("total : ", len(os.listdir(f"{root_dir}/{from_data_dir}/train/{data_type}")) + \
                        len(os.listdir(f"{root_dir}/{from_data_dir}/valid/{data_type}")) + \
                        len(os.listdir(f"{root_dir}/{from_data_dir}/test/{data_type}")))

# 데이터셋을 한 곳에 모을 공간에서의 데이터 수 체크
def checkToDatasetCount(root_dir, to_data_dir, data_type) :
    print("train : ", len(os.listdir(f"{root_dir}/{to_data_dir}/train/{data_type}")))
    print("valid : ", len(os.listdir(f"{root_dir}/{to_data_dir}/valid/{data_type}")))
    print("test : ", len(os.listdir(f"{root_dir}/{to_data_dir}/test/{data_type}")))

    print("total : ", len(os.listdir(f"{root_dir}/{to_data_dir}/train/{data_type}")) + \
                        len(os.listdir(f"{root_dir}/{to_data_dir}/valid/{data_type}")) + \
                        len(os.listdir(f"{root_dir}/{to_data_dir}/test/{data_type}")))

# data(image, label) 파일 복사
def copyfile(root_dir, to_dir, split, version_num, data_type) :
    from_dir = f'DMD.v{version_num}i.yolov11'
    from_file = glob.glob(f'{root_dir}/{from_dir}/{split}/{data_type}/*')

    to_file = f'{root_dir}/{to_dir}/{split}/{data_type}'

    print(f"!!version '{version_num}', '{split}', '{data_type}' copy start!!")

    for files in tqdm(from_file) :
        filename = os.path.basename(files)
        shutil.copyfile(files, to_file+'/'+filename)


# data.yaml 파일 복사
def copyYaml(root_dir, to_dir) :
    from_dir = f'DMD.v7i.yolov11'

    # 원본 yaml 파일 복사
    shutil.copyfile(f'{root_dir}/{from_dir}/data.yaml', f'{root_dir}/{to_dir}/data.yaml')

    # data yaml 파일 설정
    with open(f'{root_dir}/{to_dir}/data.yaml', 'r') as f :
        data = yaml.full_load(f)

    data['train'] = f'{root_dir}/{to_dir}/train/images/'
    data['val'] = f'{root_dir}/{to_dir}/valid/images/'
    data['test'] = f'{root_dir}/{to_dir}/test/images/'

    with open(f'{root_dir}/{to_dir}/data.yaml', 'w') as f :
        yaml.dump(data, f)

    with open(f'{root_dir}/{to_dir}/data.yaml', 'r') as f :
        data = yaml.full_load(f)

    print(data)