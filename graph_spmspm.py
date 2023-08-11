import os
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

dir = './graph_set'
file_list = os.listdir(dir)
file_list.sort()
sp_file = None
for i, file in enumerate(file_list):
    if '64' not in file and '128' not in file and '512' not in file:
        sp_file = file
        print(sp_file)
        os.system('./spgemm %s' % (os.path.join(dir, sp_file)))