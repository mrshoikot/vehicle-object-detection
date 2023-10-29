from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tensorflow as tf

import easyocr
from PIL import Image
print(tf.__version__)

img = 'output/car/7_0.9581449627876282.jpeg'


# Load the image

image = Image.open(img)
image = image.rotate(-90) 
image.show()

image_np = np.array(image)

reader = easyocr.Reader(['bn','en']) # this needs to run only once to load the model into memory
result = reader.readtext(image_np)
#bounds = reader.readtext(img, detail=0,paragraph=True, y_ths=0.1, x_ths=0.1,text_threshold=0.9)#showonyltext
#print(result)
#print(bounds)

print(result[0],result[1])


# # Define bounding box coordinates
a_min = [result[0][0][0][0],result[0][0][0][1]]
a_max = [result[0][0][2][0],result[0][0][2][1]]
b_min = [result[1][0][0][0],result[1][0][0][1]]
b_max = [result[1][0][2][0],result[1][0][2][1]]
print(a_min,a_max,b_min,b_max)

# # Add text
textA = result[0][1]
textB = result[1][1]
#font_path = "E:\Work\python\AI\easyocr\SolaimanLipi.ttf"  # Replace with your font path
font_path = "E:\Work\python\AI\easyocr\kalpurush.ttf"  # Replace with your font path





# Create a drawing object
draw = ImageDraw.Draw(image)


# Draw bounding box
box_color = (0, 255, 0)  # Green color (RGB)
box_thickness = 2
boxA = [(a_min[0], a_min[1]), (a_max[0], a_max[1])]
boxB = [(b_min[0], b_min[1]), (b_max[0], b_max[1])]
print(boxA,boxB)
draw.rectangle(boxA, outline=box_color, width=box_thickness)
draw.rectangle(boxB, outline=box_color, width=box_thickness)

# Add text
font_size = 40
font_color = (255, 255, 255)  # White color (RGB)
#font_path = "E:\Work\python\AI\easyocr\SolaimanLipi.ttf"  # Replace with your font path
#font_path = "E:\Work\python\AI\easyocr\SutonnyOMJ.ttf"  # Replace with your font path
font = ImageFont.truetype(font_path, font_size, encoding='unic')
text_x = a_min[0]
text_y = a_min[1] - font_size - 15  # Adjust to position the text above the bounding box
draw.text((text_x, text_y), textA, fill=font_color, font=font)  # Use the font object here
text_x = b_max[0]
text_y = b_max[1] - font_size - 15  # Adjust to position the text above the bounding box
draw.text((text_x, text_y), textB, fill=font_color, font=font)  # Use the font object here


# Save or display the modified image
image.show()
image.save('output_image1.jpg')  # Uncomment this line to save the modified image