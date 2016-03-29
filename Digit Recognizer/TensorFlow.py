import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
IMAGE_TO_DISPLAY = 10
data = pd.read_csv('train.csv')
#print ('data({0[0]},{0[1]})'.format(data.shape))
#print (data.head())
images = data.iloc[:,1:].values
images = images.astype(np.float)
images = np.multiply(images,1.0/255.0)

print ('images({0[0]},{0[1]})'.format(images.shape))
image_size = images.shape[1]
print ('image_size => {0}'.format(image_size))
image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)
print ('image_width => {0}\nimage_height => {1}'.format(image_width,image_height))

def display(img):
    one_image = img.reshape(image_width,image_height)
    plt.axis('off')
    plt.imshow(one_image)

display(images[IMAGE_TO_DISPLAY])