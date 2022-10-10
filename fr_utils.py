# -*- coding: utf-8 -*-
"""
Some utils for image loading and processing

@author: iurii
"""


import skimage.transform


def image_crop_center(img,crop_persentage):
    
    x,y,c = img.shape
    wdth_new = int(x*max(0,min(crop_persentage,1)))
    hght_new = int(y*max(0,min(crop_persentage,1)))
    startx = (x-wdth_new)//2
    starty = (y-hght_new)//2   
    return img[startx:startx+wdth_new,starty:starty+hght_new]


def image_normalize(image, sub_mean = [0.0,0.0,0.0], norm_val = 1.0):
    return (image-sub_mean)/norm_val

def image_preprocessed(dataset_path = "", 
                       file_name = "",
                       im_size = (224,224,3), 
                       norm_val = 1, 
                       cropp_percentage = 1.0, 
                       sub_mean =[0,0,0]):

    return image_normalize(
                skimage.transform.resize(
                    image_crop_center(
                        load_image(dataset_path = dataset_path, 
                                   file_name = file_name), 
                        cropp_percentage),
                    im_size),
                sub_mean = sub_mean, 
                norm_val = norm_val)