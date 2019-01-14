import airsim #pip install airsim
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

filename = "C:/temp/"


#convert to hsl
def to_hsl(img):
	return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

#convert to grayscale
def grayscale(img):
	return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#convert to yellow 
def isolate_yellow_hsl(img):
	low_threshold = np.array([15, 38, 115], dtype = np.uint8)
	high_threshold = np.array([35, 204, 255], dtype = np.uint8)

	yello_mask = cv2.inRange(img, low_threshold, high_threshold)
	return yello_mask
#convert to white image
def isolate_white_hsl(img):
	low_threshold = np.array([0, 200, 0], dtype = np.uint8)
	high_threshold = np.array([180, 255, 255], dtype = np.uint8)

	white_mask = cv2.inRange(img, low_threshold, high_threshold)
	return white_mask
#or yellow and white image af ther  and with original image
def combine_hsl_isolated_with_original(img, hsl_yellow, hsl_white):
	hsl_mask = cv2.bitwise_or(hsl_yellow, hsl_white)

	return cv2.bitwise_and(img, img, mask=hsl_mask)

#filter image
def filter_img_hsl(img):
	hsl_img = to_hsl(img)
	hsl_yellow = isolate_yellow_hsl(hsl_img)
	hsl_white = isolate_white_hsl(hsl_img)

	return combine_hsl_isolated_with_original(img, hsl_yellow, hsl_white)

#gaussian blur algorithm
def gaussian_blur(grayscale_img, kernel_size=3):
	return cv2.GaussianBlur(grayscale_img, (kernel_size, kernel_size), 0)

#get edge from image
def canny_edge_detector(blurred_img, low_threshold, high_threshold):
	return cv2.Canny(blurred_img, low_threshold, high_threshold)


def get_vertices_for_img(img):
    imshape = img.shape
    lower_left = [imshape[1]/40, imshape[0]]
    print("This is lower left", lower_left)
    lower_right = [imshape[1] - imshape[1]/40, imshape[0]]
    print("This is lower right", lower_right)
    top_left = [imshape[1]/2 - imshape[1]/2, imshape[0]/2 + imshape[0]/10]
    print("This is top left", top_left)
    top_right = [imshape[1]/2 + imshape[1]/2, imshape[0]/2 + imshape[0]/10]
    print("This is top right", top_right)
    vert =[np.array([lower_left, top_left, lower_right, top_right ], dtype=np.int32)]
    print("This is vertices", vert)
    return vert

def region_of_interest(img):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    vert = get_vertices_for_img(img)
    print("This is mask", mask)
    print("This is ignore mask color", ignore_mask_color)
    cv2.fillPoly(mask, vert, ignore_mask_color)
    
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def hough_transform(segmented_img, rho, theta, threshold, min_line_len, max_line_gap):
    return cv2.HoughLinesP(segmented_img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

def draw_lines(img, lines, color=[255, 0, 0], thickness=10, make_copy=True):
    img_copy = np.copy(img) if make_copy else img
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img_copy, (x1, y1), (x2, y2), color, thickness)
            
    return img_copy

def detectLane(img):
	
	rho = 1
	theta = (np.pi/180) * 1
	threshold = 15
	min_line_length = 20
	max_line_gap = 10
	
	img_filter = filter_img_hsl(img)
	grayscale_img = grayscale(img_filter)
	blurred_img = gaussian_blur(grayscale_img, kernel_size = 5)
	canny_img = canny_edge_detector(blurred_img, 50, 150)
	segmented_img = region_of_interest(canny_img)
	hough_lines_img = hough_transform(segmented_img, rho, theta, threshold, min_line_length, max_line_gap)
	img_with_lines = draw_lines(img, hough_lines_img)

	return img_with_lines

def controlCar(client):
	car_controls = airsim.CarControls()
	car_controls.throttle = 0.5
	car_controls.steering = 0
	client.setCarControls(car_controls)

def main():
	client = airsim.CarClient()
	client.confirmConnection()
	client.enableApiControl(True)

	controlCar(client)
	import time
	while True:
		responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
		for response in responses:
			start = time.time()
			img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) 
			img_rgba = img1d.reshape(response.height, response.width, 4)  
	#		img_rgba = np.flipud(img_rgba)
	#		airsim.write_png(os.path.normpath(filename + 'inputDetectLine3.png'), img_rgba) 
	#		img_with_lines = detectLane(img_rgba)
			img_rgba = cv2.cvtColor(img_rgba, cv2.COLOR_BGRA2BGR)
			cv2.imshow("Display image", img_rgba)
			print("The processing time", time.time() - start)
			if cv2.waitKey(1) & 0xFF == ord('q'):
					break
	client.enableApiControl(False)
#			cv2.imshow('Display window', img_with_lines)
#			if cv2.waitKey(1) & 0xFF == ord('q'):
#				break
#	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()