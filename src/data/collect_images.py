##################################################
# Script to collect all images in the same folder#
##################################################

import shutil
import os
import glob

dst_dir = "data/external/images/all"
src_dir = "data/external/images/Images/"

if not os.path.exists(dst_dir):
    os.mkdir(dst_dir)

for dirpath, dirnames, filenames in os.walk(src_dir):
    for i in range(len(filenames)):
        shutil.copy(dirpath + '/' + filenames[i], dst_dir)
