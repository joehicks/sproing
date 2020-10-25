# ML Builder
**DMS Hackathon 2020** entry for **Postman Pasty**.

## Setup venv
In the downloaded directory setup a python virtual environment and install the dependencies inside it
```sh
python3 -m venv venv
source venv/bin/activate
pip install tensorflow Pillow opencv-python
```

## Build a config file
1. Navigate to http://joehicks.co.uk/dms2020
2. Drag in an image taken by the camera you wish to use for analysis
3. Click two points to set the cropping boundary
4. Click additional points to set NOK target areas to classify
5. Copy and paste the config.json output to the config.json in the downloaded directory

## Prepare your foldeer structure
With the config file in place run ./build_folders.py to build a folder structure

## Populate folders with images
Place images for each condition in the folders in ./source_images

## Convert images
Run ./scale_images to crop your images

## Train model
Run ./train_model.py to train the ML image classification model

## Run model
Run ./predict.py path/to/test/image.jpg to run a prediction on an image
```sh
python predict_springs.py path/to/image.jpg
```