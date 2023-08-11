import os
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

dir = './cnn_set'
sub_dir = ['alexnet', 'resnet18', 'vgg16']
for i in range(3):
    dir_name = os.path.join(dir, sub_dir[i], 'weight')
    feature_dir = os.path.join(dir, sub_dir[i], 'feature')
    file_list = os.listdir(dir_name)
    file_list.sort()
    feature_list = os.listdir(feature_dir)
    feature_list.sort()
    for i, file in enumerate(file_list):
        if 'mtx' in file and 'dense' not in file:
            weight_file = file
            print(weight_file)
            feature_file = feature_list[i]
            print(feature_file)
            os.system('./spmspm %s %s' % (os.path.join(dir_name, weight_file), os.path.join(feature_dir, feature_file)))