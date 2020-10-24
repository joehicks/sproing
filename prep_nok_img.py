from PIL import Image
import pathlib
import random


img_dir = pathlib.Path("images")
k = 0

img1 = Image.open("images/HLA_L.png")
img1 = img1.convert("RGB")
for k in range(1, 120):
    v_shift = random.randint(-20, 20)
    h_shift = random.randint(-20, 20)
    rot = random.randint(-5, 5)
    img = img1.rotate(rot)
    img = img.crop((470 + h_shift, 300 + v_shift, 1170 +h_shift, 800 + v_shift))
    img = img.resize((175, 125))
    img.save("prepped/train/hla_l/%04dhlal.jpg" % k)

    """

for i in list(img_dir.glob("ok/*.bmp")):
    img1 = Image.open(str(i))
    img = img1.crop((470, 300, 1170, 800))
    img = img.resize((175, 125))
    img.save("prepped/train/ok/%04d.jpg" % k)

    v_shift = random.randint(-5, 5)
    h_shift = random.randint(-5, 5)
    rot = random.randint(-1, 1)
    img = img1.rotate(rot)
    img = img.crop((500 + h_shift, 300 + v_shift, 1200 +h_shift, 700 + v_shift))
    img = img.resize((175, 100))
    img.save("img4/train/ok/%04da.jpg" % k)
    k+=1

    """