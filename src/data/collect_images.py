##################################################
# Script to collect all images in the same folder#
##################################################

import shutil
import os

dst_dir = "data/external/images/all"
src_dir = "data/external/images/Images/"

if not os.path.exists(dst_dir):
    os.mkdir(dst_dir)
    
# loops through folders
for dirpath, dirnames, filenames in os.walk(src_dir):
    # looops through images
    for i in range(len(filenames)):
        shutil.copy(dirpath + '/' + filenames[i], dst_dir)
