import os
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# dir = './cnn_set'
# sub_dir = ['alexnet', 'resnet18', 'vgg16']
# for i in range(3):
#     dir_name = os.path.join(dir, sub_dir[i], 'weight')
#     file_list = os.listdir(dir_name)
#     file_list.sort()
#     sp_file = None
#     for i, file in enumerate(file_list):
#         if 'mtx' in file and 'dense' not in file:
#             sp_file = file
#             print(sp_file)
#             os.system('./spmm %s %d' % (os.path.join(dir_name, sp_file), 64))
#             os.system('./spmm %s %d' % (os.path.join(dir_name, sp_file), 128))
#             os.system('./spmm %s %d' % (os.path.join(dir_name, sp_file), 512))

import scanpy as sc
# adata = sc.read('cnn_set/alexnet/feature/features.8_feature.mtx')
# wdata = sc.read('cnn_set/alexnet/weight/features.8_weight.mtx')
# adata = sc.read('cnn_set/resnet18/feature/layer3.1.conv1_feature_batch.mtx')
# wdata = sc.read('cnn_set/resnet18/weight/layer3.1.conv1_weight.mtx')
# adata = sc.read('cnn_set/resnet18/feature/layer4.1.conv2_feature_batch.mtx')
# wdata = sc.read('cnn_set/resnet18/weight/layer4.1.conv2_weight.mtx')
adata = sc.read('cnn_set/vgg16/feature/features.27_feature.mtx')
wdata = sc.read('cnn_set/vgg16/weight/features.27_weight.mtx')
adata = adata.X
wdata = wdata.X
print(adata.shape)
print(wdata.shape)


