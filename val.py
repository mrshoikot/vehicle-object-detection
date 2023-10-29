from ultralytics import YOLO
from matplotlib import pyplot as plt
from PIL import Image
import os
import glob

# Create save directory if it doesn't exist
if not os.path.exists('output'):
    os.mkdir('output')

# Load a model
# model = YOLO('yolov8m.pt')  # load an official model
model = YOLO('./best.pt')  # load a custom model


# Run batched inference on a list of images
results = model.track("IMG_1038.MOV", stream=True,show=True, device="mps", conf=0.5)  # return a generator of Results objects


# Process results generator
for result in results:
    # Get all bounding boxes
    boxes = result.boxes

    # Get all class names as dict
    names = result.names

    # Original image for this frame as numpy array
    image = result.orig_img

   


    for box in boxes:
        # Get class name for this box
        cls = names[int(box.cls)]

        # Get coordinates of box
        coords = box.xyxy[0]

        # Crop box from image using coordinates
        item = image[int(coords[1]):int(coords[3]), int(coords[0]):int(coords[2])]

        # Create save directory if it doesn't exist
        if not os.path.exists('output/' + cls):
            os.mkdir('output/' + cls)

        # Check if the item is already saved using the box id
        files = glob.glob("output/" + cls + "/" + str(int(box.id)) + "_*")

        shouldSave = False
        if len(files) == 0:
            # This item was not saved before, so it should be saved
            shouldSave = True
        else:
            # This item was already saved, check if the confidence is higher
            current_conf = float(files[0].split("/")[-1].split("_")[1].split(".jpeg")[0])
            if float(box.conf) > current_conf:
                shouldSave = True

                # Delete the old file
                os.remove(files[0])

        if shouldSave:
            im = Image.fromarray(item)
            filename = "output/" + cls + "/" + str(int(box.id)) + "_" + str(float(box.conf)) + ".jpeg"
            im.save(filename)

        
        