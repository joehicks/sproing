import os
import json
import pathlib

with open("config.json") as f:
    config = json.load(f)

scale = config["scale"]
left = int(float(config["crop"]["x"]) * scale)
top = int(float(config["crop"]["y"]) * scale)
right = int(float(left) * scale) + config["crop"]["w"]
bottom = int(float(top) * scale) + config["crop"]["h"]

w = right - left
h = bottom - top


os.makedirs("prepared/train/ok")
os.makedirs("prepared/valid/ok")

image_dir = pathlib.Path("source_images/ok")
k = 0
for img1 in list(image_dir.glob("*")):
    img = Image.open(str(img))
    img = img1.crop((left, top, right, bottom))
    if k < 18:
        img.save("prepared/valid/ok/%04d.jpg" % k)
    else:
        img.save("prepared/train/ok/%04d.jpg" % k)
    k += 1

for nok in config["noks"]:
    os.makedirs("prepared/train/" + nok["name"])
    os.makedirs("prepared/valid/" + nok["name"])
    image_dir = pathlib.Path("source_images/" + nok["name"])
    k = 0
    for img1 in list(image_dir.glob("*")):
        img = Image.open(str(img))
        img = img1.crop((left, top, right, bottom))
        if k < 18:
            img.save("prepared/valid/" + nok["name"] + "/%04d.jpg" % k)
        else:
            img.save("prepared/train/" + nok["name"] + "/%04d.jpg" % k)
        k += 1
    
print("images scaled, please run ./train_model.py")