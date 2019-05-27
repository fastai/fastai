import cv2
import os
import re
import numpy as np

class_list = []
color_dic = dict()
flag = 0

def color_gen():
    '''
    Generate a new color. As first color generates (0, 255, 0)
    '''
    global flag
  
    if flag == 0:
        color = (0, 255, 0)
        flag += 1
    else:
        np.random.seed()
        color = tuple(255 * np.random.rand(3))
    return color

def show(class_name, download_dir, label_dir,total_images, index):
    '''
    Show the images with the labeled boxes.

    :param class_name: self explanatory
    :param download_dir: folder that contains the images
    :param label_dir: folder that contains the labels
    :param index: self explanatory
    :return: None
    '''
 
    global class_list, color_dic

    if not os.listdir(download_dir)[index].endswith('.jpg'):
        index += 2
    img_file = os.listdir(download_dir)[index]
    current_image_path = str(os.path.join(download_dir, img_file))
    img = cv2.imread(current_image_path)
    file_name = str(img_file.split('.')[0]) + '.txt'
    file_path = os.path.join(label_dir, file_name)
    f = open(file_path, 'r')

    window_name = "Visualizer: {}/{}".format(index+1, total_images)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    width = 500
    height = int((img.shape[0] * width) / img.shape[1])
    cv2.resizeWindow(window_name, width, height)

    for line in f:        
        # each row in a file is class_name, XMin, YMix, XMax, YMax
        match_class_name = re.compile('^[a-zA-Z]+(\s+[a-zA-Z]+)*').match(line)
        class_name = line[:match_class_name.span()[1]]
        ax = line[match_class_name.span()[1]:].lstrip().rstrip().split(' ')
	# opencv top left bottom right

        if class_name not in class_list:
            class_list.append(class_name)
            color = color_gen()     
            color_dic[class_name] = color  

        font = cv2.FONT_HERSHEY_SIMPLEX
        r ,g, b = color_dic[class_name]
        cv2.putText(img,class_name,(int(float(ax[0]))+5,int(float(ax[1]))-7), font, 0.8,(b, g, r), 2,cv2.LINE_AA)
        cv2.rectangle(img, (int(float(ax[-2])), int(float(ax[-1]))),
                      (int(float(ax[-4])),
                       int(float(ax[-3]))), (b, g, r), 3)

    cv2.imshow(window_name, img)
