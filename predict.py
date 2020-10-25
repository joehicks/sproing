import numpy as np
import os
from PIL import Image
import tensorflow as tf
import pathlib
import cv2
import json

from tensorflow.keras import layers
import sys

img_path = sys.argv[1]

with open("config.json") as f:
    config = json.load(f)

scale = config["scale"]
left = int(float(config["crop"]["x"]) * scale)
top = int(float(config["crop"]["y"]) * scale)
right = int(float(left) * scale) + config["crop"]["w"]
bottom = int(float(top) * scale) + config["crop"]["h"]

w = right - left
h = bottom - top

img = Image.open(img_path)
img = img1.crop((left, top, right, bottom))
img.save("working_image.jpg")

checkpoint_path = "checkpoint/model.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

latest = tf.train.latest_checkpoint(checkpoint_dir)

model = tf.keras.Sequential([
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(len(config["noks"]) + 1)
])

model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model.load_weights(latest)

img = tf.keras.preprocessing.image.load_img(
  "working_image.jpg"
)
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) 


predictions = model.predict(
  img_array
)
score = tf.nn.softmax(predictions[0])

"""
print(score)
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
"""

ok = np.argmax(score) == 2


print("{},{:.5e},{:.5e},{:.5e},{:.5e},{:.5e}".format("OK" if ok else "NOK", score[2], score[3], score[4], score[0], score[1]))