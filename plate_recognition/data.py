import torch
from torch.utils.data import Dataset
import glob
import os
from torchvision import transforms
from pathlib import Path
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        # 获取到当前路径下的所有车牌路径
        self.file_list = glob.glob(os.path.join(self.path, '*.*'))

    def __getitem__(self, item):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 168))
        ])
        path = Path(self.file_list[item]).resolve()
        label = path.stem.split('_')[0]
        img = Image.open(path)
        img = transform(img)

        return img, label

    def __len__(self):
        return len(self.file_list)


if __name__ == '__main__':
    a = CustomDataset()
    print(a[0])
