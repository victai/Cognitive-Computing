import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
from pycocotools.coco import COCO
import ipdb

class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, json, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        print('json', json)
        self.coco = COCO(json)
        
        all_object_ids = [i['id'] for i in self.coco.dataset['categories']]
        tmp = {i['id']: i for i in self.coco.dataset['categories']}
        self.object_id_mapping = {k: v for k, v in zip(all_object_ids, range(3, len(all_object_ids)+3))}
        self.object_id_mapping['<PAD>'] = 0
        self.object_id_mapping['<START>'] = 1
        self.object_id_mapping['<END>'] = 2
        self.inverse_object_id_mapping = {v: k for k, v in self.object_id_mapping.items()}
        self.inverse_object_id_mapping[0] = {'id': "<PAD>", 'supercategory': "<PAD>", 'name': '<PAD>'}
        self.inverse_object_id_mapping[1] = {'id': "<START>", 'supercategory': "<START>", 'name': '<START>'}
        self.inverse_object_id_mapping[2] = {'id': "<END>", 'supercategory': '<END>', 'name': '<END>'}
        for k, v in self.inverse_object_id_mapping.items():
            if k >= 3:
                self.inverse_object_id_mapping[k] = tmp[v]


        self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        img_id = self.ids[index]
        object_ids = [self.object_id_mapping[k['category_id']] for k in coco.imgToAnns[img_id]]
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        object_ids = [1] + object_ids + [2]
        target = torch.LongTensor(object_ids)
        return image, target

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, objects = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(obj) for obj in objects]
    targets = torch.zeros(len(objects), max(lengths)).long()
    for i, obj in enumerate(objects):
        end = lengths[i]
        targets[i, :end] = obj[:end]        
    return images, targets, lengths

def get_loader(root, json, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(root=root,
                       json=json,
                       transform=transform)
    
    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn,
                                              drop_last=True)
    return data_loader
