""" a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py """

import fire
import os
import lmdb
import cv2

import numpy as np


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)

def createDataset(inputPath, gtFile, outputPath, checkValid=True):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=1 * 1024 * 1024 * 1024)
    cache = {}
    cnt = 1

    with open(gtFile, 'r') as data:
        datalist = data.readlines()

    nSamples = len(datalist) // 2
    for i in range(0, len(datalist), 2):
        imagePathLine = datalist[i].strip().replace("Filename: ", "")
        labelLine = datalist[i + 1].strip()

        # # only use alphanumeric data
        # if re.search('[^a-zA-Z0-9]', label):
        #     continue

        imagePath = os.path.join(inputPath, imagePathLine)
        label = labelLine

        if not os.path.exists(imagePath):
            print(f'{imagePath} does not exist')
            continue

        with open(imagePath, 'rb') as f:
            imageBin = f.read()

        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print(f'{imagePath} is not a valid image')
                    continue
            except Exception as e:
                print(f'Error occurred with {i//2}th image: {e}')
                with open(outputPath + '/error_image_log.txt', 'a') as log:
                    log.write(f'{i//2}-th image data occurred error\n')
                continue

            imageKey = f'image-{cnt:09d}'.encode()
            labelKey = f'label-{cnt:09d}'.encode()
            cache[imageKey] = imageBin
            cache[labelKey] = label.encode('utf-8')

            if cnt % 1000 == 0:
                writeCache(env, cache)
                cache = {}
                print(f'Written {cnt} / {nSamples}')

            cnt += 1

    nSamples = cnt - 1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print(f'Created dataset with {nSamples} samples')


if __name__ == '__main__':
    fire.Fire(createDataset)
