import glob
import os
import shutil

root_dir = '/data0/SISR_dataset/LSDIR/LR/train'
dst_dir = '/data0/SISR_dataset/LSDIR/LR/x4'
for fname in sorted(os.listdir(root_dir)):
    path = os.path.join(root_dir, fname)
    if os.path.isdir(path):
        for img in os.listdir(path):
            cur_file = os.path.join(path, img)
            shutil.move(cur_file, dst_dir)
            # shutil.rmtree(path)
