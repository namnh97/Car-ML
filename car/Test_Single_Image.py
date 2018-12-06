# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 21:48:30 2018

@author: namnh997
"""
import airsim #pip install airsim
import numpy as np
import os
import matplotlib.pyplot as plt

filename = "C:/temp/"

client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])

response = responses[0]

# get numpy array
img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) 

# reshape array to 4 channel image array H X W X 4
img_rgba = img1d.reshape(response.height, response.width, 4)  

# original image is fliped vertically
img_rgba = np.flipud(img_rgba)
 
print(img_rgba)
# just for fun add little bit of green in all pixels

# write to png 
airsim.write_png(os.path.normpath(filename + 'inputDetectLine.png'), img_rgba) 


