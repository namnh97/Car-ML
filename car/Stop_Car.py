# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 21:09:51 2018

@author: namnh997
"""

import airsim


client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
    
car_controls = airsim.CarControls()
car_controls.throttle = 0
car_controls.steering = 0
client.setCarControls(car_controls)

client.enableApiControl(False)
