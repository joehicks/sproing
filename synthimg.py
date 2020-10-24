from PIL import Image
import pathlib
import random



img_dir = pathlib.Path("imgbmp/ok")
j = 0
for i in list(img_dir.glob("*.bmp")):
    for k in range(1, 240):
        v_shift = random.randint(-20, 20)
        h_shift = random.randint(-20, 20)
        rot = random.randint(-5, 5)
        img = Image.open(str(i))
        img = img.rotate(rot)
        img = img.crop((500 + h_shift, 300 + v_shift, 1200 +h_shift, 700 + v_shift))
        img = img.resize((175, 100))
        if (j < 30):
            img.save(("img4/valid/nok/%03d"% (k)) + ".jpg")
        else:
            img.save(("img4/train/nok/%03d"% (k)) + ".jpg")
        j += 1

