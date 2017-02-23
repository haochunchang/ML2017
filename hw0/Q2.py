from PIL import Image, ImageChops
import numpy as np
import sys

im1 = Image.open(sys.argv[1])
im2 = Image.open(sys.argv[2])
width, height = im1.size
result = np.zeros((height, width, 4), dtype=np.uint8) 

for i in range(width):
    for j in range(height):
        
        if im1.getpixel((i,j)) != im2.getpixel((i,j)):
            result[j][i] = im2.getpixel((i,j))
            
result = result.astype(np.uint8)
result = Image.fromarray(result, "RGBA")
result.save("ans_two.png")

