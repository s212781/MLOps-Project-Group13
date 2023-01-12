#############################################################################
# Script to crop the images so that only the dogs will be shown in the data #
#############################################################################

from PIL import Image
import os
import cv2

# To open image
path = "data/external/images/all/"

# Create new folder
save_dir = "data/external/images/all_cropped"
try:
    os.mkdir(save_dir)
except:
    pass

# loops through folders
for filename in os.listdir(path):
    name = filename
    img = cv2.imread(path+name)
    
    # Finding the matching xml file
    breed_id =  name[name.find('n0'):name.find('_')]
    xml_id = breed_id + '_' + name[name.find('_')+1:name.find('.')]

    # The xml_directory
    xml_dir = "data/external/annotations/Annotation"

    # this is the folder of the dog breed
    xml_folder = [name for name in os.listdir(xml_dir) if breed_id in name][0]

    final_path = xml_dir+ '/' + xml_folder+ '/' + xml_id

    # C:\Users\thorl\Documents\DTU\JAN23\MLOps-Project-Group13\data\external\annotations\Annotation\n02085620-Chihuahua\n02085620_7
    with open(final_path,'r') as xml_file:
        # xml_file.read()
        lines = xml_file.readlines()

        for line in lines:
            if line.find('bndbox') != -1:
                idx = lines.index(line)
                break
        strng = []
        #finding the lines describing the bounding box
        for i in range(4):
            strng.append(lines[idx+i+1])

    # assigning values
    xmin = int(strng[0][strng[0].find('<xmin>')+6:strng[0].find('</xmin>')])
    xmax = int(strng[2][strng[2].find('<xmax>')+6:strng[2].find('</xmax>')])
    ymin = int(strng[1][strng[1].find('<ymin>')+6:strng[1].find('</ymin>')])
    ymax = int(strng[3][strng[3].find('<ymax>')+6:strng[3].find('</ymax>')])

    #cropping image
    crop_img = img[ymin:ymax,xmin:xmax]
    #saving new image
    cv2.imwrite(save_dir+'/'+name, crop_img)
