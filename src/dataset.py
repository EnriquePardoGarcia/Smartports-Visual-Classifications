import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torch
from torchvision import transforms


class PortDataset(Dataset):
    def __init__(self, csv_path, images_dir, transform=None):
        self.data = pd.read_csv(csv_path, sep=None, engine="python")
        self.images_dir = images_dir
        self.transform = transform
        self.image_col = self.data.columns[0]
        self.label_col = self.data.columns[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx][self.image_col]
        label = int(self.data.iloc[idx][self.label_col])
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(augment=False, image_size=224, aug_level=1.0):
    """
    aug_level: 0.0-1.0, controla la intensidad del augmentation
    1.0 = augmentation completo, 0.5 = moderado, etc.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5 * aug_level),
            transforms.RandomVerticalFlip(p=0.2 * aug_level),
            transforms.ColorJitter(
                brightness=0.4 * aug_level,
                contrast=0.4 * aug_level,
                saturation=0.3 * aug_level,
                hue=0.05 * aug_level
            ),
            transforms.RandomGrayscale(p=0.05 * aug_level),
            transforms.RandomRotation(degrees=int(15 * aug_level)),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, val_transform


def build_dataloaders(csv_path, images_dir, augment=False, val_split=0.2,
                      batch_size=32, image_size=224, seed=42, aug_level=1.0):
    train_tf, val_tf = get_transforms(augment=augment, image_size=image_size, aug_level=aug_level)

    full_dataset = PortDataset(csv_path, images_dir, transform=None)
    n_total = len(full_dataset)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(full_dataset, [n_train, n_val], generator=generator)

    train_dataset = PortDataset(csv_path, images_dir, transform=train_tf)
    val_dataset = PortDataset(csv_path, images_dir, transform=val_tf)

    train_final = Subset(train_dataset, train_subset.indices)
    val_final = Subset(val_dataset, val_subset.indices)

    train_loader = DataLoader(train_final, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_final, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)
    return train_loader, val_loader
