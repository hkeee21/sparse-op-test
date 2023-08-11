import scanpy as sc
import torch
import time
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def mm_test(A, B):
    for _ in range(10):
        _ = torch.matmul(A, B)

    for _ in range(100):
        torch.cuda.synchronize()
        st = time.time()
        _ = torch.matmul(A, B)
        torch.cuda.synchronize()
        ed = time.time()
    
    return (ed - st) * 1000

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

# A_addr = 'cnn_set/alexnet/weight/features.8_weight_dense_256_3456.mtx'
# B_addr = 'cnn_set/alexnet/feature/features.8_feature_dense_3456_169.mtx'

# A_addr = 'cnn_set/resnet18/weight/layer3.1.conv1_weight_dense_256_2304.mtx'
# B_addr = 'cnn_set/resnet18/feature/layer3.1.conv1_feature_batch_dense_2304_245.mtx'

# A_addr = 'cnn_set/resnet18/weight/layer4.1.conv2_weight_dense_512_4608.mtx'
# B_addr = 'cnn_set/resnet18/feature/layer4.1.conv2_feature_batch_dense_4608_196.mtx'

# A_addr = 'cnn_set/vgg16/weight/features.17_weight_dense_256_2304.mtx'
# B_addr = 'cnn_set/vgg16/feature/features.17_feature_dense_2304_3136.mtx'

# A_addr = 'cnn_set/vgg16/weight/features.40_weight_dense_512_4608.mtx'
# B_addr = 'cnn_set/vgg16/feature/features.40_feature_dense_4608_196.mtx'

A_addr = 'cnn_set/vgg16/weight/features.27_weight_dense_512_4608.mtx'
B_addr = 'cnn_set/vgg16/feature/features.27_feature_dense_4608_784.mtx'

adata = sc.read(B_addr)
wdata = sc.read(A_addr)
adata = adata.X
wdata = wdata.X
print(adata.shape)
print(wdata.shape)

A = torch.rand(wdata.shape, dtype=torch.float32, device='cuda')
B = torch.rand(adata.shape, dtype=torch.float32, device='cuda')

dur = mm_test(A, B)

print("%.4f ms" % dur)