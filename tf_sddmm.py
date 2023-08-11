import os
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

dir = './tf_set'
file_list = os.listdir(dir)
file_list.sort()
sp_file = None
for i, file in enumerate(file_list):
    if '512' not in file and '1024' not in file:
        sp_file = file
        print(sp_file)
        os.system('./sddmm %s %d' % (os.path.join(dir, sp_file), 512))
        os.system('./sddmm %s %d' % (os.path.join(dir, sp_file), 1024))