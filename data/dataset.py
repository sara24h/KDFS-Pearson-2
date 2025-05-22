import csv
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import random
from torchdata.datapipes.iter import IterableWrapper, FileOpener, Mapper
import pickle
import hashlib

class FaceDataset(Dataset):
    def __init__(self, data, root_dir, transform=None, img_column='path', cache_dir=None, use_cache=True, cache_threshold=50000):
        self.data = data  # List of (image_path, label) tuples
        self.root_dir = root_dir
        self.transform = transform
        self.img_column = img_column
        self.label_map = {1: 1, 0: 0, 'real': 1, 'fake': 0, 'Real': 1, 'Fake': 0}
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.cache_threshold = cache_threshold
        self.cache = {}  # In-memory cache

        # Initialize cache
        if self.use_cache and len(self.data) <= self.cache_threshold:
            print(f"Using in-memory cache for dataset with {len(self.data)} samples")
            self._initialize_memory_cache()
        elif self.use_cache and cache_dir:
            print(f"Using disk cache for dataset with {len(self.data)} samples at {cache_dir}")
            os.makedirs(cache_dir, exist_ok=True)

    def _initialize_memory_cache(self):
        """Initialize in-memory cache for small datasets"""
        for idx in range(len(self.data)):
            img_path, _ = self.data[idx]
            full_path = os.path.join(self.root_dir, img_path)
            if os.path.exists(full_path):
                image = Image.open(full_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                self.cache[idx] = image

    def _get_cache_key(self, img_path):
        """Generate a unique cache key for an image"""
        return hashlib.md5(img_path.encode()).hexdigest()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        full_path = os.path.join(self.root_dir, img_path)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image not found: {full_path}")

        # Check in-memory cache
        if self.use_cache and idx in self.cache:
            image = self.cache[idx]
        else:
            # Check disk cache
            if self.use_cache and self.cache_dir:
                cache_key = self._get_cache_key(full_path)
                cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
                if os.path.exists(cache_path):
                    with open(cache_path, 'rb') as f:
                        image = pickle.load(f)
                else:
                    image = Image.open(full_path).convert('RGB')
                    if self.transform:
                        image = self.transform(image)
                    # Save to disk cache
                    with open(cache_path, 'wb') as f:
                        pickle.dump(image, f)
            else:
                image = Image.open(full_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)

            # Store in in-memory cache if small dataset
            if self.use_cache and len(self.data) <= self.cache_threshold:
                self.cache[idx] = image

        label = self.label_map[label]
        return image, torch.tensor(label, dtype=torch.float)

class Dataset_selector:
    def __init__(
        self,
        dataset_mode,  # 'hardfake', 'rvf10k', or '140k'
        hardfake_csv_file=None,
        hardfake_root_dir=None,
        rvf10k_train_csv=None,
        rvf10k_valid_csv=None,
        rvf10k_root_dir=None,
        realfake140k_train_csv=None,
        realfake140k_valid_csv=None,
        realfake140k_test_csv=None,
        realfake140k_root_dir=None,
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=8,
        pin_memory=True,
        ddp=False,
        cache_dir='/tmp/dataset_cache',  # Directory for disk cache
    ):
        if dataset_mode not in ['hardfake', 'rvf10k', '140k']:
            raise ValueError("dataset_mode must be 'hardfake', 'rvf10k', or '140k'")

        self.dataset_mode = dataset_mode

        # Define image size based on dataset_mode
        image_size = (256, 256) if dataset_mode in ['rvf10k', '140k'] else (300, 300)

        # Define mean and std based on dataset_mode
        if dataset_mode == 'hardfake':
            mean = (0.5124, 0.4165, 0.3684)
            std = (0.2363, 0.2087, 0.2029)
        elif dataset_mode == 'rvf10k':
            mean = (0.5212, 0.4260, 0.3811)
            std = (0.2486, 0.2238, 0.2211)
        else:  # dataset_mode == '140k'
            mean = (0.5207, 0.4258, 0.3806)
            std = (0.2490, 0.2239, 0.2212)

        # Split transforms into random and non-random
        non_random_transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        random_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(image_size[0], padding=8),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.8, 1.2)),
        ])

        transform_train = transforms.Compose([non_random_transforms, random_transforms])
        transform_test = non_random_transforms

        # Function to read CSV with csv module
        def read_csv_file(csv_file, path_column='path', label_column='label'):
            data = []
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data.append((row[path_column], row[label_column]))
            return data

        # Function to create image paths for hardfake
        def create_image_path(images_id, label):
            folder = 'fake' if label.lower() == 'fake' else 'real'
            img_name = os.path.basename(images_id)
            if not img_name.endswith('.jpg'):
                img_name += '.jpg'
            return os.path.join(folder, img_name)

        # Load data based on dataset_mode
        if dataset_mode == 'hardfake':
            if not hardfake_csv_file or not hardfake_root_dir:
                raise ValueError("hardfake_csv_file and hardfake_root_dir must be provided")
            data = []
            with open(hardfake_csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    img_path = create_image_path(row['images_id'], row['label'])
                    data.append((img_path, row['label']))
            root_dir = hardfake_root_dir

            train_data, temp_data = train_test_split(
                data, test_size=0.3, stratify=[x[1] for x in data], random_state=3407
            )
            val_data, test_data = train_test_split(
                temp_data, test_size=0.5, stratify=[x[1] for x in temp_data], random_state=3407
            )

        elif dataset_mode == 'rvf10k':
            if not rvf10k_train_csv or not rvf10k_valid_csv or not rvf10k_root_dir:
                raise ValueError("rvf10k_train_csv, rvf10k_valid_csv, and rvf10k_root_dir must be provided")
            train_data = read_csv_file(rvf10k_train_csv, path_column='path', label_column='label')
            valid_data = read_csv_file(rvf10k_valid_csv, path_column='path', label_column='label')
            val_data, test_data = train_test_split(
                valid_data, test_size=0.5, stratify=[x[1] for x in valid_data], random_state=3407
            )
            root_dir = rvf10k_root_dir

        else:  # dataset_mode == '140k'
            if not realfake140k_train_csv or not realfake140k_valid_csv or not realfake140k_test_csv or not realfake140k_root_dir:
                raise ValueError("realfake140k_train_csv, realfake140k_valid_csv, realfake140k_test_csv, and realfake140k_root_dir must be provided")
            train_data = read_csv_file(realfake140k_train_csv, path_column='path', label_column='label')
            val_data = read_csv_file(realfake140k_valid_csv, path_column='path', label_column='label')
            test_data = read_csv_file(realfake140k_test_csv, path_column='path', label_column='label')
            root_dir = os.path.join(realfake140k_root_dir, 'real_vs_fake', 'real-vs-fake')
            random.seed(3407)
            random.shuffle(train_data)
            random.shuffle(val_data)
            random.shuffle(test_data)

        # Debug: Print data statistics
        from collections import Counter
        print(f"{dataset_mode} dataset statistics:")
        print(f"Sample train image paths: {[x[0] for x in train_data[:5]]}")
        print(f"Total train dataset size: {len(train_data)}")
        print(f"Train label distribution: {Counter([x[1] for x in train_data])}")
        print(f"Sample validation image paths: {[x[0] for x in val_data[:5]]}")
        print(f"Total validation dataset size: {len(val_data)}")
        print(f"Validation label distribution: {Counter([x[1] for x in val_data])}")
        print(f"Sample test image paths: {[x[0] for x in test_data[:5]]}")
        print(f"Total test dataset size: {len(test_data)}")
        print(f"Test label distribution: {Counter([x[1] for x in test_data])}")

        # Check for missing images
        for split, data in [('train', train_data), ('validation', val_data), ('test', test_data)]:
            missing_images = [img_path for img_path, _ in data if not os.path.exists(os.path.join(root_dir, img_path))]
            if missing_images:
                print(f"Missing {split} images: {len(missing_images)}")
                print(f"Sample missing {split} images: {missing_images[:5]}")

        # Create datasets with caching
        train_dataset = FaceDataset(
            train_data, root_dir, transform=transform_train, img_column='path', 
            cache_dir=os.path.join(cache_dir, 'train'), use_cache=True
        )
        val_dataset = FaceDataset(
            val_data, root_dir, transform=transform_test, img_column='path', 
            cache_dir=os.path.join(cache_dir, 'val'), use_cache=True
        )
        test_dataset = FaceDataset(
            test_data, root_dir, transform=transform_test, img_column='path', 
            cache_dir=os.path.join(cache_dir, 'test'), use_cache=True
        )

        # Create data loaders with DDP support
        if ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
            
            self.loader_train = DataLoader(
                train_dataset, batch_size=train_batch_size, num_workers=num_workers,
                pin_memory=pin_memory, sampler=train_sampler,
            )
            self.loader_val = DataLoader(
                val_dataset, batch_size=eval_batch_size, num_workers=num_workers,
                pin_memory=pin_memory, sampler=val_sampler,
            )
            self.loader_test = DataLoader(
                test_dataset, batch_size=eval_batch_size, num_workers=num_workers,
                pin_memory=pin_memory, sampler=test_sampler,
            )
        else:
            self.loader_train = DataLoader(
                train_dataset, batch_size=train_batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=pin_memory,
            )
            self.loader_val = DataLoader(
                val_dataset, batch_size=eval_batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
            )
            self.loader_test = DataLoader(
                test_dataset, batch_size=eval_batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
            )

        # Debug: Print loader sizes
        print(f"Train loader batches: {len(self.loader_train)}")
        print(f"Validation loader batches: {len(self.loader_val)}")
        print(f"Test loader batches: {len(self.loader_test)}")

        # Test a sample batch
        for loader, name in [(self.loader_train, 'train'), (self.loader_val, 'validation'), (self.loader_test, 'test')]:
            try:
                sample = next(iter(loader))
                print(f"Sample {name} batch image shape: {sample[0].shape}")
                print(f"Sample {name} batch labels: {sample[1]}")
            except Exception as e:
                print(f"Error loading sample {name} batch: {e}")

if __name__ == "__main__":
    # Example for hardfakevsrealfaces
    dataset_hardfake = Dataset_selector(
        dataset_mode='hardfake',
        hardfake_csv_file='/kaggle/input/hardfakevsrealfaces/data.csv',
        hardfake_root_dir='/kaggle/input/hardfakevsrealfaces',
        train_batch_size=64,
        eval_batch_size=64,
        ddp=True,
        cache_dir='/tmp/hardfake_cache',
    )

    # Example for rvf10k
    dataset_rvf10k = Dataset_selector(
        dataset_mode='rvf10k',
        rvf10k_train_csv='/kaggle/input/rvf10k/train.csv',
        rvf10k_valid_csv='/kaggle/input/rvf10k/valid.csv',
        rvf10k_root_dir='/kaggle/input/rvf10k',
        train_batch_size=64,
        eval_batch_size=64,
        ddp=True,
        cache_dir='/tmp/rvf10k_cache',
    )

    # Example for 140k Real and Fake Faces
    dataset_140k = Dataset_selector(
        dataset_mode='140k',
        realfake140k_train_csv='/kaggle/input/140k-real-and-fake-faces/train.csv',
        realfake140k_valid_csv='/kaggle/input/140k-real-and-fake-faces/valid.csv',
        realfake140k_test_csv='/kaggle/input/140k-real-and-fake-faces/test.csv',
        realfake140k_root_dir='/kaggle/input/140k-real-and-fake-faces',
        train_batch_size=64,
        eval_batch_size=64,
        ddp=True,
        cache_dir='/tmp/140k_cache',
    )
