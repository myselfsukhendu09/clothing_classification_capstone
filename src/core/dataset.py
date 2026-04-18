import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

class ClothingDataPipeline:
    """Refined and Modular Data Pipeline for Architectural Deployment."""
    
    def __init__(self, data_root, image_size=224, batch_size=24):
        self.data_root = data_root
        self.image_size = image_size
        self.batch_size = batch_size
        
        # 1. Advanced Augmentation Profile
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2),
        ])

        # Standard Evaluation Profile
        self.val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_dataloaders(self):
        """Constructing high-throughput Loaders for CV Optimization."""
        
        if not os.path.exists(self.data_root):
            raise FileNotFoundError(f"Dataset root {self.data_root} not found.")
            
        # Automatic class mapping via folder topology
        full_dataset = datasets.ImageFolder(self.data_root)
        idx_to_label = {v: k for k, v in full_dataset.class_to_idx.items()}
        
        # Split (70-15-15)
        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.15 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size

        train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size])
        
        # Apply transformation profiles
        train_ds.dataset.transform = self.train_transform
        val_ds.dataset.transform = self.val_transform
        test_ds.dataset.transform = self.val_transform

        # Optimized Loaders
        loader_kwargs = {
            'batch_size': self.batch_size,
            'num_workers': 4,
            'pin_memory': True,
            'persistent_workers': True
        }

        train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
        test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

        return train_loader, val_loader, test_loader, idx_to_label

def get_dataloaders(csv_file=None, img_dir=None, batch_size=24, image_size=224):
    """Refined Legacy Wrapper for Synchronization."""
    # Note: csv_file is ignored for more robust ImageFolder topology
    pipeline = ClothingDataPipeline(data_root=img_dir, image_size=image_size, batch_size=batch_size)
    return pipeline.get_dataloaders()
