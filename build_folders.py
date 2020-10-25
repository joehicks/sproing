import os
import json
with open("config.json") as f:
    config = json.load(f)

    os.makedirs("source_images/ok")
for nok in config["noks"]:
    os.makedirs("source_images/" + nok["name"])
    print(nok["name"])

print("folders created, please add your images to the ./source_images folder structure and then run ./scale_images.py")