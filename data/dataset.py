import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data.distributed import DistributedSampler
import glob
import argparse

class FaceDataset(Dataset):
    def __init__(self, data_frame, root_dir, transform=None, img_column='filename'):
        self.data = data_frame
        self.root_dir = root_dir
        self.transform = transform

        # Check for appropriate column for image paths
        possible_columns = ['filename', 'image', 'path', 'original_path', 'images_id']
        for col in possible_columns:
            if col in self.data.columns:
                self.img_column = col
                break
        else:
            raise ValueError(f"No image column found. Expected columns: {possible_columns}")

        # Label mapping
        self.label_map = {1: 1, 0: 0, 'real': 1, 'fake': 0, 'Real': 1, 'Fake': 0, 'ai': 0}

        # Check for existence of image files
        missing_images = []
        for img_path in self.data[self.img_column]:
            full_path = os.path.join(self.root_dir, img_path.lstrip('/'))
            if not os.path.exists(full_path):
                missing_images.append(full_path)
        if missing_images:
            print(f"Warning: {len(missing_images)} images not found in {self.root_dir}")
            print("Sample missing images:", missing_images[:5])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[self.img_column].iloc[idx].lstrip('/')
        img_name = os.path.join(self.root_dir, img_path)
        if not os.path.exists(img_name):
            print(f"Warning: Image not found: {img_name}, returning None")
            return None, None  # Handle this in the training loop
        image = Image.open(img_name).convert('RGB')
        label = self.label_map.get(self.data['label'].iloc[idx], self.data['label'].iloc[idx])
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float)

class Dataset_selector:
    def __init__(
        self,
        dataset_mode,
        realfake125k_root_dir=None,
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=8,
        pin_memory=True,
        ddp=False,
    ):
        if dataset_mode != '125k':
            raise ValueError("This script only supports dataset_mode='125k'")

        self.dataset_mode = dataset_mode

        # Set image size for 125k dataset
        image_size = (160, 160)

        # Set mean and standard deviation for normalization
        mean = (0.3822, 0.3073, 0.2586)
        std = (0.2124, 0.2033, 0.1806)

        # Define data transformations
        transform_train = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(image_size[0], padding=8),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        # Select appropriate column for image paths
        img_column = 'filename'

        # Load data for 125k dataset
        if not realfake125k_root_dir:
            raise ValueError("realfake125k_root_dir must be provided")
        root_dir = realfake125k_root_dir

        def create_dataframe(split):
            data = {'filename': [], 'label': []}
            possible_splits = [split, split.capitalize()]
            real_path = None
            fake_path = None
            for s in possible_splits:
                # Check for lowercase folder names explicitly since dataset uses 'real' and 'fake'
                temp_real = os.path.join(root_dir, s, 'real')
                temp_fake = os.path.join(root_dir, s, 'fake')
                print(f"Checking split: {s}, Real folder: real, Fake folder: fake")
                print(f"Real path: {temp_real}, Exists: {os.path.exists(temp_real)}")
                print(f"Fake path: {temp_fake}, Exists: {os.path.exists(temp_fake)}")
                if os.path.exists(temp_real) and os.path.exists(temp_fake):
                    real_path = temp_real
                    fake_path = temp_fake
                    break
            else:
                raise FileNotFoundError(f"No valid split directory found for {split} in {root_dir}")

            print(f"Selected real path: {real_path}")
            print(f"Selected fake path: {fake_path}")

            for ext in ['*.jpg', '*.jpeg', '*.png']:
                real_images = glob.glob(os.path.join(real_path, ext))
                fake_images = glob.glob(os.path.join(fake_path, ext))
                print(f"Found {len(real_images)} real images in {real_path} with extension {ext}")
                print(f"Found {len(fake_images)} fake images in {fake_path} with extension {ext}")

                for img_path in real_images:
                    data['filename'].append(os.path.relpath(img_path, root_dir))
                    data['label'].append(1)  # Real = 1
                for img_path in fake_images:
                    data['filename'].append(os.path.relpath(img_path, root_dir))
                    data['label'].append(0)  # Fake = 0

            df = pd.DataFrame(data)
            if df.empty:
                raise ValueError(f"No images found in {split} directory")
            return df

        train_data = create_dataframe('train')
        val_data = create_dataframe('validation')
        # If test directory doesn't exist, split validation data for test
        try:
            test_data = create_dataframe('test')
        except FileNotFoundError:
            print("Warning: Test directory not found, splitting validation data for test")
            val_data, test_data = train_test_split(
                val_data, test_size=0.5, stratify=val_data['label'], random_state=3407
            )

        train_data = train_data.sample(frac=1, random_state=3407).reset_index(drop=True)
        val_data = val_data.sample(frac=1, random_state=3407).reset_index(drop=True)
        test_data = test_data.sample(frac=1, random_state=3407).reset_index(drop=True)

        # Reset indices
        train_data = train_data.reset_index(drop=True)
        val_data = val_data.reset_index(drop=True)
        test_data = test_data.reset_index(drop=True)

        # Print dataset samples for debugging
        print(f"Train data sample:\n{train_data.head()}")
        print(f"Validation data sample:\n{val_data.head()}")
        print(f"Test data sample:\n{test_data.head()}")

        # Calculate and print dataset split percentages
        total_images = len(train_data) + len(val_data) + len(test_data)
        train_percent = (len(train_data) / total_images) * 100 if total_images > 0 else 0
        val_percent = (len(val_data) / total_images) * 100 if total_images > 0 else 0
        test_percent = (len(test_data) / total_images) * 100 if total_images > 0 else 0

        print(f"{dataset_mode} dataset split percentages:")
        print(f"Training: {train_percent:.2f}% ({len(train_data)} images)")
        print(f"Validation: {val_percent:.2f}% ({len(val_data)} images)")
        print(f"Test: {test_percent:.2f}% ({len(test_data)} images)")

        # Print dataset statistics
        print(f"{dataset_mode} dataset statistics:")
        print(f"Sample train image paths:\n{train_data[img_column].head()}")
        print(f"Total train dataset size: {len(train_data)}")
        print(f"Train label distribution:\n{train_data['label'].value_counts()}")
        print(f"Sample validation image paths:\n{val_data[img_column].head()}")
        print(f"Total validation dataset size: {len(val_data)}")
        print(f"Validation label distribution:\n{val_data['label'].value_counts()}")
        print(f"Sample test image paths:\n{test_data[img_column].head()}")
        print(f"Total test dataset size: {len(test_data)}")
        print(f"Test label distribution:\n{test_data['label'].value_counts()}")

        # Create datasets
        train_dataset = FaceDataset(train_data, root_dir, transform=transform_train, img_column=img_column)
        val_dataset = FaceDataset(val_data, root_dir, transform=transform_test, img_column=img_column)
        test_dataset = FaceDataset(test_data, root_dir, transform=transform_test, img_column=img_column)

        # Set up DataLoader
        if ddp:
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
            val_sampler = DistributedSampler(val_dataset, shuffle=True)
            test_sampler = DistributedSampler(test_dataset, shuffle=False)

            self.loader_train = DataLoader(
                train_dataset,
                batch_size=train_batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                sampler=train_sampler,
            )
            self.loader_val = DataLoader(
                val_dataset,
                batch_size=eval_batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                sampler=val_sampler,
            )
            self.loader_test = DataLoader(
                test_dataset,
                batch_size=eval_batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                sampler=test_sampler,
            )
        else:
            self.loader_train = DataLoader(
                train_dataset,
                batch_size=train_batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            self.loader_val = DataLoader(
                val_dataset,
                batch_size=eval_batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            self.loader_test = DataLoader(
                test_dataset,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

        print(f"Number of train batches: {len(self.loader_train)}")
        print(f"Number of validation batches: {len(self.loader_val)}")
        print(f"Number of test batches: {len(self.loader_test)}")

        # Check sample batches with label distribution
        for loader, name in [(self.loader_train, 'train'), (self.loader_val, 'validation'), (self.loader_test, 'test')]:
            try:
                sample = next(iter(loader))
                if sample[0] is None or sample[1] is None:
                    print(f"Warning: Sample {name} batch contains None values")
                else:
                    print(f"Sample {name} batch image shape: {sample[0].shape}")
                    print(f"Sample {name} batch labels: {sample[1]}")
                    print(f"{name} batch label distribution: {torch.bincount(sample[1].int())}")
            except Exception as e:
                print(f"Error loading sample {name} batch: {e}")

if __name__ == "__main__":
    dataset_hardfake = Dataset_selector(
        dataset_mode='hardfake',
        hardfake_csv_file='/kaggle/input/hardfakevsrealfaces/data.csv',
        hardfake_root_dir='/kaggle/input/hardfakevsrealfaces',
        train_batch_size=256,
        eval_batch_size=128,
        ddp=True,
    )

    dataset_rvf10k = Dataset_selector(
        dataset_mode='rvf10k',
        rvf10k_train_csv='/kaggle/input/rvf10k/train.csv',
        rvf10k_valid_csv='/kaggle/input/rvf10k/valid.csv',
        rvf10k_root_dir='/kaggle/input/rvf10k',
        train_batch_size=256,
        eval_batch_size=128,
        ddp=True,
    )

    dataset_140k = Dataset_selector(
        dataset_mode='140k',
        realfake140k_train_csv='/kaggle/input/140k-real-and-fake-faces/train.csv',
        realfake140k_valid_csv='/kaggle/input/140k-real-and-fake-faces/valid.csv',
        realfake140k_test_csv='/kaggle/input/140k-real-and-fake-faces/test.csv',
        realfake140k_root_dir='/kaggle/input/140k-real-and-fake-faces',
        train_batch_size=256,
        eval_batch_size=128,
        ddp=True,
    )

    dataset_200k = Dataset_selector(
        dataset_mode='200k',
        realfake200k_train_csv='/kaggle/input/200k-real-vs-ai-visuals-by-mbilal/train_labels.csv',
        realfake200k_val_csv='/kaggle/input/200k-real-vs-ai-visuals-by-mbilal/val_labels.csv',
        realfake200k_test_csv='/kaggle/input/200k-real-vs-ai-visuals-by-mbilal/test_labels.csv',
        realfake200k_root_dir='/kaggle/input/200k-real-vs-ai-visuals-by-mbilal/my_real_vs_ai_dataset',
        train_batch_size=128,
        eval_batch_size=128,
        ddp=True,
    )

    dataset_190k = Dataset_selector(
        dataset_mode='190k',
        realfake190k_root_dir='/kaggle/input/deepfake-and-real-images',
        train_batch_size=128,
        eval_batch_size=128,
        ddp=True,
    )

    dataset_12_9k = Dataset_selector(
        dataset_mode='12.9k',
        dataset_12_9k_csv_file='/kaggle/input/deepfake-face-images/dataset.csv',
        dataset_12_9k_root_dir='/kaggle/input/deepfake-face-images',
        train_batch_size=128,
        eval_batch_size=128,
        ddp=True,
    )

    dataset_330k = Dataset_selector(
        dataset_mode='330k',
        realfake330k_root_dir='/kaggle/input/deepfake-dataset',
        train_batch_size=128,
        eval_batch_size=128,
        ddp=True,
    )

    dataset_125k = Dataset_selector(
        dataset_mode=args.dataset_mode,
        realfake125k_root_dir=args.data_dir,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        ddp=True,
    )
