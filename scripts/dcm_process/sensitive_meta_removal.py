import SimpleITK as sitk
import numpy as np
import os
import json
from sensitive_meta_policy import sensitive_tag_matching_policy


source_root='/mnt/e/Workspace/Telesurgery/Data/Field Data/2023_11_10'
target_root='/mnt/e/Workspace/Telesurgery/Data/Field Data/2023_11_10_processed'

available_extensions = ['.dcm', '.DCM', '.ima', '.IMA']


def recursive_process(source_root, target_root, process_func):
    removed_list =[]
    for file in os.listdir(source_root):
        source_path = os.path.join(source_root, file)
        if os.path.isdir(source_path):
            target_path = os.path.join(target_root, file)
            recursive_mkdir(target_path)
            removed_list.extend(recursive_process(source_path, target_path, process_func))
        else:
            if os.path.splitext(file)[1] in available_extensions:
                print('Processing: ', source_path)
                removed_list.append({ source_path: process_func(file, source_path, target_root) })
    return removed_list
    
    
def remove_sensitive_meta(filename, file_path, target_path)-> map :
    print('Processing: ', file_path)
    raw = sitk.ReadImage(file_path)
    removed = {}
    for key in raw.GetMetaDataKeys():
        if match_any(key, sensitive_tag_matching_policy):
            value = raw.GetMetaData(key)
            print('     Removing: ', key, '->', value)
            if (raw.EraseMetaData(key)):
                removed[key] = value
    sitk.WriteImage(raw, os.path.join(target_path, filename), imageIO="GDCMImageIO")
    return removed


def match_any(key, policies) -> bool:
    for policy in policies:
        if match(key, policy):
            return True
    return False

def match(key, policy) -> bool:
    if 'group_number' in policy:
        return key[0:4] == policy['group_number']
    elif 'full_tag' in policy:
        expected = policy['full_tag'][0] + '|' + policy['full_tag'][1]
        return key == expected
    else:
        raise Exception(f'Invalid policy: {policy}')
    
def recursive_mkdir(target_path):
    if os.path.exists(os.path.dirname(target_path)):
        if not os.path.exists(target_path):
            os.mkdir(target_path)
        return
    else:
        recursive_mkdir(os.path.dirname(target_path))

    
if __name__ == '__main__':
    removed_list = recursive_process(source_root, target_root, remove_sensitive_meta)
    summary = { 'root': source_root, 'removed_metadata': removed_list }
    json.dump(summary, open(os.path.join(target_root, 'summary.json'), 'w'), indent=4)