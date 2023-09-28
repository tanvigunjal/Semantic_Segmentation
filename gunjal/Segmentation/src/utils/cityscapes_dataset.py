import os
import numpy as np
import scipy.misc as m
import torch
from torch.utils.data import Dataset


def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
    :param rootdir is the root directory
    :param suffix is the suffix to be searched
    :return: all the files matching the criteria
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]


class Cityscape_dataset(Dataset):
    color_map = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    # make a dict for mapping. Example: 0 -> [128,64,128]
    color_map_dict = dict(zip(range(19), color_map))

    def __init__(
        self,
        root,
        split="train",                  # which data split to use
        is_transform=True,              # transform function activation
        img_size=(512, 1024),           # image_size to use in transform function
    ):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.n_classes = 19
        self.files = {}

        self.images_base = os.path.join(self.root, "leftImg8bit", self.split)
        self.annotations_base = os.path.join(self.root, "gtFine", self.split)
        
        # contains list of all pngs inside all different folders. Recursively iterates 
        self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".png")

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33,]

        self.class_names = [
            "unlabelled",
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]

        self.ignore_index = 250

        # dictionary of valid classes 7:0, 8:1, 11:2
        self.class_map = dict(zip(self.valid_classes, range(19)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))
        
        # prints number of images found
        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])
    
    # returns image and label
    def __getitem__(self, index):
        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
        )

        # read images 
        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        # read labels
        lbl = m.imread(lbl_path)
        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl
    
    # transform function
    def transform(self, img, lbl):
        # resize image
        img = m.imresize(img, (self.img_size[0], self.img_size[1]))
        img = img[:, :, ::-1] # RGB -> BGR
        img = img.astype(np.float64)
        img = img.transpose(2, 0, 1)

        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), "nearest", mode="F")
        lbl = lbl.astype(int)

        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")

        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
            print("after det", np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")
        
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl
    
    # encode segmentation map
    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        
        # Put all valid classes to one
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        
        return mask
    
    # decode segmentation map
    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()

        for l in range(0, self.n_classes):
            r[temp == l] = self.color_map[l][0]
            g[temp == l] = self.color_map[l][1]
            b[temp == l] = self.color_map[l][2]
        
        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0

        return rgb


def dataloader(config):
    train_data = Cityscape_dataset(config.dataset_path, split="train")
    val_data = Cityscape_dataset(config.dataset_path, split="val")
    test_data = Cityscape_dataset(config.dataset_path, split="test")

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=config.batch_size, 
        shuffle=True, num_workers=config.num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=config.batch_size, 
        shuffle=True, num_workers=config.num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=config.batch_size, 
        shuffle=True, num_workers=config.num_workers
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    dataset_path = "/ds/images/Cityscapes/"
    train_loader, val_loader, test_loader = dataloader(dataset_path)

    print(len(train_loader))
    print(len(val_loader))
    print(len(test_loader))

    for i, (images, labels) in enumerate(train_loader):
        print(images.shape)
        print(labels.shape)
        break