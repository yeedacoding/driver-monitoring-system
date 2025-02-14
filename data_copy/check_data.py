from data4yolo import checkFromDatasetCount, checkToDatasetCount

org_data_root = '/Volumes/G-DRIVE USB'
to_data_dir = 'dataset'

version_num = ['3', '4', '5', '6', '7']

# 원본 데이터의 각 버전별 개수 파악
for version in version_num :
    print("==============================")
    print(f"원본 데이터 verison {version}, 'images'")
    checkFromDatasetCount(org_data_root, version, "images")

# 복사 완료된 경로의 데이터 총 갯수 파악
print("==============================")
print("복사 완료 데이터 총 합")
checkToDatasetCount(org_data_root, to_data_dir, "images")