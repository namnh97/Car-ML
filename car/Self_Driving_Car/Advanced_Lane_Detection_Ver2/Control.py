import airsim


client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = airsim.CarControls()

car_state = client.getCarState()
print("Speed %d, Gear %d" % (car_state.speed, car_state.gear))
# go forward

def goForward():
    car_controls.throttle = 0.5
    car_controls.steering = 0
    #car_controls.is_manual_gear = True
    #car_controls.manual_gear = -1
    client.setCarControls(car_controls)

def turnRight():
    car_controls.throttle = 0.5
    car_controls.steering = 1
    #car_controls.is_manual_gear = True
    #car_controls.manual_gear = -1
    client.setCarControls(car_controls)

def turnLeft():
    car_controls.throttle = 0.5
    car_controls.steering = -1
    #car_controls.is_manual_gear = true
    #car_controls.manual_gear = -1
    client.setCarControls(car_controls)

def test():
    car_controls.throttle = 0.5
    car_controls.steering = 0
    #car_controls.is_manual_gear = true
    #car_controls.manual_gear = -1
    car_controls.brake = 0 #stop 
    client.setCarControls(car_controls)


if __name__ == "__main__":
    test()