import os
from typing import List, Tuple

import torch
import torch.utils.data as data
import torchvision
from PIL import Image
import pandas as pd

class ArtLoader(data.Dataset):
    class_labels = {
        "abstract_painting" : 0, 
        "cityscape" : 1, 
        "genre_painting" : 2,
        "illustration" : 3,
        "landscape" : 4,
        "nude_painting" : 5,
        "portrait" : 6,
        "religious_painting" : 7,
        "sketch_and_study" : 8,
        "still_life" : 9
    }

    def __init__(self, root_dir: str, split: str = "train", train_file: str = "genre_train.csv", test_file: str  = "genre_val.csv", transform: torchvision.transforms.Compose = None):
        self.root = os.path.expanduser(root_dir)
        self.transform = transform
        self.split = split

        if split == "train":
            self.curr_file = os.path.join(root_dir, train_file)
        elif split == "test":
            self.curr_file = os.path.join(root_dir, test_file)

        self.class_dict = self.class_labels
        self.dataset = self.art_with_genres()

    def art_with_genres(self) -> List[Tuple[str, int]]:
        img_paths = []
        img_paths = pd.read_csv(self.curr_file)
        img_paths.columns = ["Path", "Label"]
        img_paths["Path"].astype(str)
        img_paths["Label"].astype(int)
        return img_paths

    def art_from_path(self, path: str) -> Image:
        if os.path.exists(path):
            return Image.open(path)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img_path = self.dataset.iloc[index, 0]
        class_idx = self.dataset.iloc[index, 1]
        img = self.art_from_path(os.path.join(self.root,"wikiart", img_path))
        if self.transform:
            img = self.transform(img)
        return img, class_idx

    def __len__(self) -> int:
        return len(self.dataset)