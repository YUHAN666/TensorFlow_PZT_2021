import os
import sys

from PIL import Image


# 将输入路径的上两级路径加入系统
def set_projectpath(current_path):
    curPath = os.path.abspath(current_path)
    # curPath = os.path.abspath(os.path.dirname(__file__))
    rootPath = os.path.split(curPath)[0]
    sys.path.append(rootPath)
    rootPath = os.path.split(rootPath)[0]
    sys.path.append(rootPath)


def concatImage(images, mode="L"):
    if not isinstance(images, list):
        raise Exception('images must be a  list  ')
    count = len(images)
    size = Image.fromarray(images[0]).size
    target = Image.new(mode, (size[0] * count, size[1] * 1))
    for i in range(count):
        image = Image.fromarray(images[i]).resize(size, Image.BILINEAR)
        target.paste(image, (i * size[0], 0, (i + 1) * size[0], size[1]))
    return target


def shape2d(a):
    """
    Ensure a 2D shape.
    Args:
        a: a int or tuple/list of length 2
    Returns:
        list: of length 2. if ``a`` is a int, return ``[a, a]``.
    """
    if type(a) == int:
        return [a, a]
    if isinstance(a, (list, tuple)):
        assert len(a) == 2
        return list(a)
    raise RuntimeError("Illegal shape: {}".format(a))
