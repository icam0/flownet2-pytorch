import mmcv
from skimage import io
import os
root = './correspondence_bg/pred_C/'
files_list = []

for path, subdirs, files in os.walk(root):
    for name in files:
        files_list.append(os.path.join(path, name))
files_list = sorted(files_list)
print(files_list)
files_list = [file for file in files_list if 'DS_Store' not in file]
print(files_list)
for file_name in files_list:
    flow = mmcv.flow2rgb(mmcv.flowread(file_name))
    io.imsave(file_name[0:-3]+'png', flow)