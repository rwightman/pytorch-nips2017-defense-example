import torch.utils.data as data

from PIL import Image
import os
import os.path
import torch

IMG_EXTENSIONS = ['.png', '.jpg']


def is_image_file(filename):
    return any(filename.tolower().endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def find_inputs(folder, types=IMG_EXTENSIONS):
    inputs = []
    for root, _, files in os.walk(folder, topdown=False):
        for rel_filename in files:
            base, ext = os.path.splitext(rel_filename)
            if ext.lower() in types:
                abs_filename = os.path.join(root, rel_filename)
                inputs.append((abs_filename, None))  #FIXME support folder structures with classes as well
    return inputs


class Dataset(data.Dataset):

    def __init__(self, root, transform=None):
        classes, class_to_idx = find_classes(root)
        imgs = find_inputs(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            return img
        else:
            return img, target

    def __len__(self):
        return len(self.imgs)

    def filenames(self, indices=[]):
        if indices:
            return [self.imgs[i][0] for i in indices]
        else:
            return [x[0] for x in self.imgs]
