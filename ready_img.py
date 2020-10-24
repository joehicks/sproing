from PIL import Image
import sys

img = Image.open(sys.argv[1])
img = img.convert("RGB")
img = img.crop((470, 300, 1170, 800))
img = img.resize((175, 125))
img.save("%s.jpg" % sys.argv[2])