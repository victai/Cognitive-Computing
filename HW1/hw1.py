import matplotlib.pyplot as plt
from PIL import ImageDraw, ImageFont, Image
from scipy import ndimage
import numpy as np

# input.png source: https://i.imgur.com/O9ygWOL.jpg
# Font: DejaVuSans.ttf

img = plt.imread("input.png")
plt.imshow(img)
plt.show()

sobelx = ndimage.sobel(img, 1).clip(0,1)
sobelx = Image.fromarray(np.uint8(sobelx*255))
draw = ImageDraw.Draw(sobelx)
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)  # fontsize = 30
draw.text((img.shape[1]*0.01, img.shape[0] * 0.9), "B04902105", fill=(0,0,255), font=font)

plt.imshow((sobelx))
#plt.show()
plt.savefig("output.png")
