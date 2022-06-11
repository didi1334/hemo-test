import os.path

from PIL import Image

# path = "C:\\Users\\dia13\\OneDrive\\Рабочий стол\\hemo\\PNG"
path = "C:\\Users\\dia13\\OneDrive\\Рабочий стол\\1"
dirs = os.listdir(path)


def crop():
    for item in dirs:
        fullpath = os.path.join(path, item)  # corrected
        if os.path.isfile(fullpath):
            im = Image.open(fullpath)
            f, e = os.path.splitext(fullpath)
            imCrop = im.crop((0, 0, 51, 28))  # corrected
            imCrop.save(f + 'hemo.png', "PNG", quality=100)

crop()
