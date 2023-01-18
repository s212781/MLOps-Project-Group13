#############################################################################
# Script to crop the images so that only the dogs will be shown in the data #
#############################################################################

from PIL import Image
import os
import cv2

path = "data/external/images/Images/"
src_dir = "data/external/images"
# Create new folder
save_dir = "data/processed/images"

try:
    os.mkdir(save_dir)
except:
    pass

# loops through folders
# dirpath = data/external/images/Images/n02116738-African_hunting_dog
# filenames = nxxxxxxxxx.jpg
for dirpath, dirnames, filenames in os.walk(src_dir):
    # looops through images
    for i in range(len(filenames)):
        # Finding the matching xml file
        breed_id = dirpath[dirpath.find("n0") :]

        # create directory
        breed_folder = save_dir + "/" + breed_id
        try:
            os.mkdir(breed_folder)
        except:
            pass

        name = filenames[i]
        img = cv2.imread(dirpath + "/" + name)

        # xml_id = breed_id + '_' + name[name.find('_')+1:name.find('.')]

        # The xml_directory
        xml_dir = "data/external/annotations/Annotation"

        final_path = xml_dir + "/" + breed_id + "/" + name[: name.find(".")]

        # this is the folder of the dog breed
        # xml_folder = [name for name in os.listdir(xml_dir) if breed_id in name][0]

        # final_path = xml_dir+ '/' + xml_folder+ '/' + xml_id

        # C:\Users\thorl\Documents\DTU\JAN23\MLOps-Project-Group13\data\external\annotations\Annotation\n02085620-Chihuahua\n02085620_7
        with open(final_path, "r") as xml_file:
            # xml_file.read()
            lines = xml_file.readlines()

            for line in lines:
                if line.find("bndbox") != -1:
                    idx = lines.index(line)
                    break
            strng = []
            # finding the lines describing the bounding box
            for i in range(4):
                strng.append(lines[idx + i + 1])

        # assigning values
        xmin = int(strng[0][strng[0].find("<xmin>") + 6 : strng[0].find("</xmin>")])
        xmax = int(strng[2][strng[2].find("<xmax>") + 6 : strng[2].find("</xmax>")])
        ymin = int(strng[1][strng[1].find("<ymin>") + 6 : strng[1].find("</ymin>")])
        ymax = int(strng[3][strng[3].find("<ymax>") + 6 : strng[3].find("</ymax>")])

        # cropping image
        crop_img = img[ymin:ymax, xmin:xmax]
        # saving new image
        cv2.imwrite(breed_folder + "/" + name, crop_img)
