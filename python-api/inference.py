import cv2
import numpy as np
import json
import torch
from argparse import ArgumentParser
from glob import glob
import os
import cv2
import collections
import numbers
import numpy as np
import torch


def main(data):
        
    model = torch.jit.load('model.pth')
    decoder = json.load(open("model_decoder.json"))
    model.eval()
    latin2arabic = {'aa': 'ا', 'la': 'ل', 'sh': 'ش', 'ra': 'ر', 'ya': 'ي', 'ay': 'ع', 'da': 'د', 'wa': 'و',
                    'ta': 'ت', 'te': 'ة', 'ka': 'ق', 'sa': 'ص', 'gh': 'غ', 'ee': 'ى', 'na': 'ن',
                    'de': 'ض', 'fa': 'ف', 'ha': 'ح',
                    'ba': 'ب', 'he': 'ه', 'ze': 'ظ', 'hamala': 'لمح', 'ma': 'م',
                    'ke': 'ك', 'se': 'س', 'za': 'ظ', 'aala': 'لا',
                    'dh': 'ذ', 'ae': 'أ', 'al': 'ئ', 'aela': 'لأ',
                    'mala': 'لم', 'ah': 'إ', '7': '7', '0': '0', '2': '2',
                    'th': 'ث', 'jala': 'لج', 'hala': 'لح', 'ja': 'ج', 'hana': 'نح', '8': '8',
                    '1': '1', 'khla': 'لخ', 'heA': 'ه', '9': '9', 'kh': 'خ',
                    'to': 'ط', '6': '6', 'am': 'آ', 'ahla': 'لإ', 'sp': " ", 'hh':'ء'
                    }
    if isinstance(data, list):
        output = {'word': [], 'confidences': []}  
        for im in data:
            o = forward(im, model, latin2arabic, decoder)
            output['word'].append(o['word'])
            output['confidences'].extend(o['confidences'])
    else:
        output = forward(data, model, latin2arabic, decoder)
    return output


def forward(img, model, latin2arabic, decoder):
    img = transform_image(img)
    output = predict_probs(img, model)
    chars, confidences = probs_to_characters(output, latin2arabic, decoder)
    words = ''
    output = {'word': words.join(chars), 'confidences': confidences}
    return output

def read_image(path: str):
    if not isinstance(path, str):
        raise ValueError("Path most be str")
    elif path == "":
        raise ValueError("Empty path")
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Not Image")
    return img


def transform_image(img: np.ndarray, height: int = 32):
    if len(img.shape) == 2:
        img = np.stack((img,) * 3, axis=-1)
    elif len(img.shape) != 3:
        print("Current Image Shape:", img.shape)
        raise ValueError("Current Image Shape:", img.shape, "Expected Image of shape (height, width, channels)")
    if img.shape[2] != 3:
        raise ValueError("Expected Image to have 3 channels")
    # img = prepareImg(img)
    # img = np.stack((img,) * 3, axis=-1)
    img = pad(img, padding=20, fill=255)
    img = resize(img, height)
    img = img / 255
    # img = pad(img, padding=(4, 0), fill=255)
    img = torch.tensor(img)
    img = img.permute(2, 0, 1)
    return img.unsqueeze(0)


def predict_probs(img: torch.Tensor, model):
    return model(img.float()).squeeze(1)


def probs_to_characters(log_probs: torch.Tensor, latin2arabic, decoder):
    from pytorch_lightning.metrics import functional
    preds = functional.to_categorical(log_probs)
    out = []
    confidences = []
    for i in range(len(preds)):
        if preds[i] != 0 and preds[i] != preds[i - 1]:
            out.append(str(preds[i].item()))
            # confidences.append(torch.exp(log_probs))
            confidences.append({latin2arabic[decoder[str(preds[i].item())]]: torch.exp(torch.max(log_probs[i])).item()})
    chars = [decoder[c] for c in out]
    chars = list(reversed([latin2arabic[c] for c in chars]))
    return chars, confidences


def prepareImg(img):
    """convert given image to grayscale image (if needed) and resize to desired height"""
    assert img.ndim in (2, 3)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('1',img)
    # cv2.waitKey()
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 85, 40)
    blured1 = cv2.medianBlur(img, 3)
    blured2 = cv2.medianBlur(img, 51)
    divided = np.ma.divide(blured1, blured2).data
    normed = np.uint8(255 * divided / divided.max())
    th, threshed = cv2.threshold(normed, 100, 255, cv2.THRESH_OTSU)

    # img = np.hstack((img, blured1, blured2, normed, threshed))
    # cv2.imshow('1',img)
    # cv2.waitKey()
    return threshed

def resize(img, height):
    img = cv2.resize(img, (int(height * img.shape[1] / img.shape[0]), height))
    return img


def pad(img, padding, fill=(0, 0, 0), padding_mode='constant'):
    """Pad the given CV Image on all sides with speficified padding mode and fill value.
    Args:
        img (np.ndarray): Image to be padded.
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill (int, tuple): Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
            constant: pads with a constant value, this value is specified with fill
            edge: pads with the last value on the edge of the image
            reflect: pads with reflection of image (without repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]
            symmetric: pads with reflection of image (repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]

    Returns:
        CV Image: Padded image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be CV Image. Got {}'.format(type(img)))

    if not isinstance(padding, (numbers.Number, tuple)):
        raise TypeError('Got inappropriate padding arg')
    if not isinstance(fill, (numbers.Number, str, tuple)):
        raise TypeError('Got inappropriate fill arg')
    if not isinstance(padding_mode, str):
        raise TypeError('Got inappropriate padding_mode arg')

    if isinstance(padding, collections.Sequence) and len(padding) not in [2, 4]:
        raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                         "{} element tuple".format(len(padding)))

    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric'], \
        'Padding mode should be either constant, edge, reflect or symmetric'

    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    if isinstance(padding, collections.Sequence) and len(padding) == 2:
        pad_left = pad_right = padding[0]
        pad_top = pad_bottom = padding[1]
    if isinstance(padding, collections.Sequence) and len(padding) == 4:
        pad_left, pad_top, pad_right, pad_bottom = padding

    if isinstance(fill, numbers.Number):
        fill = (fill,) * (2 * len(img.shape) - 3)

    if padding_mode == 'constant':
        assert (len(fill) == 3 and len(img.shape) == 3) or (len(fill) == 1 and len(img.shape) == 2), \
            'channel of image is {} but length of fill is {}'.format(img.shape[-1], len(fill))

    img = cv2.copyMakeBorder(src=img, top=pad_top, bottom=pad_bottom, left=pad_left, right=pad_right,
                             borderType=PAD_MOD[padding_mode], value=fill)
    return img

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})



PAD_MOD = {'constant': cv2.BORDER_CONSTANT,
           'edge': cv2.BORDER_REPLICATE,
           'reflect': cv2.BORDER_DEFAULT,
           'symmetric': cv2.BORDER_REFLECT
           }
