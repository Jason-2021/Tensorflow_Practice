import numpy as np
import cv2 as cv
import os
from tqdm.auto import tqdm
import random

def distortion(image, x_up, x_down, amplitude, frequency, noise):
    x, y, _ = image.shape

    # distorted_image = np.copy(image)
    distorted_image = np.ones(image.shape)* 255

    for i in tqdm(range(x_up, x_down +1)):
        for j in range(y):
            y_new = j + int(amplitude * np.sin(2 * np.pi * i / frequency) + noise)
            if i == 1 and j == 0:
                print(y_new)
            #print(amplitude * np.sin(2 * np.pi * i / frequency))
            

            if 0 <= y_new < y:
                distorted_image[i, y_new] = image[i, j]
                
    
    
    cv.imwrite("aaa.png", distorted_image)


def random_distortion(image, num_area=5, y_range=None, num_waves=40, frequency_range=(20, 70)):
    '''Distort in the vertical direction'''
    x, y, _ = image.shape
    area_range = [int(i) for i in np.linspace(0, y, num_area+1)]
    
    distorted_image = np.ones(image.shape) * image[0,0]

    # num_waves =  # np.random.randint(6, 12)
    amplitude = 20 / num_waves


    # frequency_range = (20, 70)

    for area_index in range(len(area_range)-1):
        frequencies = [np.random.uniform(frequency_range[0], frequency_range[1]) for _ in range(num_waves)]
        use_cos = np.random.choice([True, False], num_waves)
        
        for i in tqdm(range(x)):
            for j in range(area_range[area_index], area_range[area_index+1]):
                wave_sum = 0

                for wave_id in range(num_waves):
                    if use_cos[wave_id]:
                        wave_sum += amplitude * np.cos(2 * np.pi * i / frequencies[wave_id])
                    else:
                        wave_sum += amplitude * np.sin(2 * np.pi * i / frequencies[wave_id])
                new_j = j + int(wave_sum)

                if 0 <= new_j < y:
                    distorted_image[i, new_j] = image[i, j]

    cv.imwrite('bbb.png', distorted_image)


if __name__ == '__main__':
    original_image = cv.imread(r'board.png')
    # print(original_image.shape)
    # os._exit(0)
    # define distortion parameters
    amplitude = 1
    frequency = 25
    noise = random.randint(-10, 10)

    x_up = 0
    x_down = 600

    # distortion(original_image, x_up, x_down, amplitude, frequency, noise)
    random_distortion(original_image)

