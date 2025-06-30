import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data.distributed import DistributedSampler
import glob

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
        hardfake_csv_file=None,
        hardfake_root_dir=None,
        rvf10k_train_csv=None,
        rvf10k_valid_csv=None,
        rvf10k_test_csv=None,
        rvf10k_root_dir=None,
        realfake140k_train_csv=None,
        realfake140k_valid_csv=None,
        realfake140k_test_csv=None,
        realfake140k_root_dir=None,
        realfake200k_train_csv=None,
        realfake200k_val_csv=None,
        realfake200k_test_csv=None,
        realfake200k_root_dir=None,
        realfake190k_root_dir=None,
        dataset_12_9k_csv_file=None,
        dataset_12_9k_root_dir=None,
        realfake330k_root_dir=None,
        realfake125k_root_dir=None,
        dataset_672k_train_label_txt=None,
        dataset_672k_val_label_txt=None,
        dataset_672k_root_dir=None,
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=8,
        pin_memory=True,
        ddp=False,
    ):
        if dataset_mode not in ['hardfake', 'rvf10k', '140k', '200k', '190k', '12.9k', '330k', '672k', '125k']:
            raise ValueError("dataset_mode must be one of 'hardfake', 'rvf10k', '140k', '200k', '190k', '12.9k', '330k', '672k', '125k'")

        self.dataset_mode = dataset_mode

        # Set image size
        if dataset_mode == '672k':
            image_size = (512, 512)
        elif dataset_mode == '125k':
            image_size = (160, 160)
        else:
            image_size = (256, 256) if dataset_mode in ['rvf10k', '140k', '200k', '190k', '12.9k', '330k'] else (300, 300)

        # Set mean and standard deviation for normalization
        if dataset_mode == 'hardfake':
            mean = (0.5124, 0.4165, 0.3684)
            std = (0.2363, 0.2087, 0.2029)
        elif dataset_mode == 'rvf10k':
            mean = (0.5214, 0.4265, 0.3814)
            std = (0.2487, 0.2240, 0.2214)
        elif dataset_mode == '140k':
            mean = (0.5207, 0.4258, 0.3806)
            std = (0.2490, 0.2239, 0.2212)
        elif dataset_mode == '200k':
            mean = (0.4868, 0.3972, 0.3624)
            std = (0.2296, 0.2066, 0.2009)
        elif dataset_mode == '190k':
            mean = (0.4668, 0.3816, 0.3414)
            std = (0.2410, 0.2161, 0.2081)
        elif dataset_mode == '12.9k':
            mean = (0.4970, 0.4026, 0.3579)
            std = (0.2579, 0.2263, 0.2190)
        elif dataset_mode == '330k':
            mean = (0.4923, 0.4042, 0.3624)
            std = (0.2446, 0.2198, 0.2141)
        elif dataset_mode == '672k':
            mean = (0.5058, 0.4068, 0.3562)
            std = (0.2583, 0.2304, 0.2226)
        elif dataset_mode == '125k':
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
        ]) if dataset_mode != '12.9k' else transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        # Select appropriate column for image paths
        img_column = 'path' if dataset_mode in ['rvf10k', '140k', '12.9k', '672k'] else 'filename' if dataset_mode in ['200k', '190k', '330k', '125k'] else 'images_id'

        # Load data based on dataset_mode
        if dataset_mode == 'rvf10k':
            if not rvf10k_train_csv or not rvf10k_valid_csv or not rvf10k_root_dir:
                raise ValueError("rvf10k_train_csv, rvf10k_valid_csv, and rvf10k_root_dir must be provided")
            print(f"Loading train CSV from: {rvf10k_train_csv}")
            print(f"Loading valid CSV from: {rvf10k_valid_csv}")
            train_data = pd.read_csv(rvf10k_train_csv)
            valid_data = pd.read_csv(rvf10k_valid_csv)
            root_dir = rvf10k_root_dir

            def create_image_path(row, split='train'):
                folder = 'fake' if row['label'] == 0 else 'real'
                img_name = os.path.basename(row['path'])
                return os.path.join('rvf10k', split, folder, img_name)

            train_data['path'] = train_data.apply(lambda row: create_image_path(row, 'train'), axis=1)
            valid_data['path'] = valid_data.apply(lambda row: create_image_path(row, 'valid'), axis=1)

            print("Sample train paths:", train_data['path'].head().tolist())
            print("Sample valid paths:", valid_data['path'].head().tolist())

            for path in train_data['path'].head():
                full_path = os.path.join(root_dir, path)
                print(f"Checking train image {full_path}: {'Exists' if os.path.exists(full_path) else 'Not Found'}")
            for path in valid_data['path'].head():
                full_path = os.path.join(root_dir, path)
                print(f"Checking valid image {full_path}: {'Exists' if os.path.exists(full_path) else 'Not Found'}")

            if rvf10k_test_csv and os.path.exists(rvf10k_test_csv):
                test_data = pd.read_csv

(rvf10k_test_csv)
                test_data['path'] = test_data.apply(lambda row: create_image_path(row, 'valid'), axis=1)
            else:
                val_data, test_data = train_test_split(
                    valid_data, test_size=0.5, stratify=valid_data['label'], random_state=3407
                )
            val_data = valid_data

        elif dataset_mode == 'hardfake':
            if not hardfake_csv_file or not hardfake_root_dir:
                raise ValueError("hardfake_csv_file and hardfake_root_dir must be provided")
            full_data = pd.read_csv(hardfake_csv_file)
            root_dir = hardfake_root_dir

            def create_image_path(row):
                folder = 'fake' if row['label'] == 'real' else 'fake'
                img_name = os.path.basename(row['path'])
                if not img_name.endswith('.jpg'):
                    img_name += '.jpg'
                return os.path.join(folder, img_name)

            full_data['path'] = full_data.apply(create_image_path, axis=1)

            train_data, temp_data = train_test_split(
                full_data, test_size=0.5, stratify=full_data['label'], random_state=3407
            )
            val_data, test_data = train_test_split(
                temp_data, test_size=0.5, stratify=temp_data['label'], random_state=3407
            )

        elif dataset_mode == '140k':
            if not realfake140k_train_csv or not realfake140k_valid_csv or not realfake140k_test_csv or not realfake140k_root_dir:
                raise ValueError("realfake140k_train_csv, realfake140k_valid_csv, realfake140k_test_csv, and realfake140k_root_dir must be provided")
            train_data = pd.read_csv(realfake140k_train_csv)
            val_data = pd.read_csv(realfake140k_valid_csv)
            test_data = pd.read_csv(realfake140k_test_csv)
            root_dir = os.path.join(realfake140k_root_dir, 'real_vs_fake', 'real-vs-fake')

            if 'path' not in train_data.columns:
                raise ValueError("CSV files for 140k dataset must contain a 'path' column")

        elif dataset_mode == '200k':
            if not realfake200k_train_csv or not realfake200k_val_csv or not realfake200k_test_csv or not realfake200k_root_dir:
                raise ValueError("realfake200k_train_csv, realfake200k_val_csv, realfake200k_test_csv, and realfake200k_root_dir must be provided")
            train_data = pd.read_csv(realfake200k_train_csv)
            val_data = pd.read_csv(realfake200k_val_csv)
            test_data = pd.read_csv(realfake200k_test_csv)
            root_dir = realfake200k_root_dir

            def create_image_path(row):
                folder = 'real' if row['label'] == 1 else 'ai_images'
                img_name = row.get('filename', row.get('image', row.get('path', '')))
                return os.path.join(folder, img_name)

            train_data['filename'] = train_data.apply(create_image_path, axis=1)
            val_data['filename'] = val_data.apply(create_image_path, axis=1)
            test_data['filename'] = test_data.apply(create_image_path, axis=1)

        elif dataset_mode == '190k':
            if not realfake190k_root_dir:
                raise ValueError("realfake190k_root_dir must be provided")
            root_dir = realfake190k_root_dir

            def create_dataframe(split):
                data = {'filename': [], 'label': []}
                real_path = os.path.join(root_dir, split, 'Real')
                fake_path = os.path.join(root_dir, split, 'Fake')

                print(f"Checking real path: {real_path}")
                print(f"Checking fake path: {fake_path}")

                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    real_images = glob.glob(os.path.join(real_path, ext))
                    fake_images = glob.glob(os.path.join(fake_path, ext))
                    print(f"Found {len(real_images)} real images in {real_path} with extension {ext}")
                    print(f"Found {len(fake_images)} fake images in {fake_path} with extension {ext}")

                    for img_path in real_images:
                        data['filename'].append(os.path.relpath(img_path, root_dir))
                        data['label'].append(1)  # Real = 1
 THEM                    for img_path in fake_images:
                        data['filename'].append(os.path.relpath(img_path, root_dir))
                        data['label'].append(0)  # Fake = 0

                df = pd.DataFrame(data)
                if df.empty:
                    raise ValueError(f"No images found in {split} directory")
                return df

            train_data = create_dataframe('Train')
            val_data = create_dataframe('Validation')
            test_data = create_dataframe('Test')

        elif dataset_mode == '12.9k':
            if not dataset_12_9k_csv_file or not dataset_12_9k_root_dir:
                raise ValueError("dataset_12_9k_csv_file and dataset_12_9k_root_dir must be provided")
            full_data = pd.read_csv(dataset_12_9k_csv_file)
            root_dir = dataset_12_9k_root_dir

            full_data['path'] = full_data['path'].str.replace('/kaggle/input/stylegan-and-stylegan2-combined-dataset/', '')
            full_data['path'] = full_data['path'].str.replace('Final Dataset/', '')

            train_data, temp_data = train_test_split(
                full_data, test_size=0.3, stratify=full_data['label'], random_state=3407
            )
            val_data, test_data = train_test_split(
                temp_data, test_size=0.5, stratify=temp_data['label'], random_state=3407
            )

        elif dataset_mode == '330k':
            if not realfake330k_root_dir:
                raise ValueError("realfake330k_root_dir must be provided")
            root_dir = realfake330k_root_dir

            def create_dataframe(split):
                data = {'filename': [], 'label': []}
                real_path = os.path.join(root_dir, split, 'Real')
                fake_path = os.path.join(root_dir, split, 'Fake')

                print(f"Checking real path: {real_path}")
                print(f"Checking fake path: {fake_path}")

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
            val_data = create_dataframe('valid')
            test_data = create_dataframe('test')

            train_data = train_data.sample(frac=1, random_state=3407).reset_index(drop=True)
            val_data = val_data.sample(frac=1, random_state=3407).reset_index(drop=True)
            test_data = test_data.sample(frac=1, random_state=3407).reset_index(drop=True)

        elif dataset_mode == '672k':
            if not dataset_672k_train_label_txt or not dataset_672k_val_label_txt or not dataset_672k_root_dir:
                raise ValueError("dataset_672k_train_label_txt, dataset_672k_val_label_txt, and dataset_672k_root_dir must be provided")
            root_dir = dataset_672k_root_dir

            def create_dataframe(split):
                if split == 'train':
                    label_file = dataset_672k_train_label_txt
                    split_dir = 'trainset'
                elif split == 'valid':
                    label_file = dataset_672k_val_label_txt
                    split_dir = 'valset'
                else:
                    raise ValueError(f"Unsupported split: {split}")

                data = {'path': [], 'label': []}
                label_path = os.path.join(root_dir, label_file)
                print(f"Loading label file for {split}: {label_path}")

                if not os.path.exists(label_path):
                    raise FileNotFoundError(f"Label file not found: {label_path}")

                with open(label_path, 'r') as f:
                    for line in f:
                        img_name, label = line.strip().split()
                        img_path = os.path.join(split_dir, img_name)
                        full_path = os.path.join(root_dir, img_path)
                        if os.path.exists(full_path):
                            data['path'].append(img_path)
                            data['label'].append(int(label))
                        else:
                            print(f"Warning: Image not found: {full_path}")

                df = pd.DataFrame(data)
                if df.empty:
                    raise ValueError(f"No valid images found in {split} directory")
                return df

            train_data = create_dataframe('train')
            val_data = create_dataframe('valid')
            test_data = val_data  # Use validation set as test set if no separate test set

        elif dataset_mode == '125k':
            if not realfake125k_root_dir:
                raise ValueError("realfake125k_root_dir must be provided")
            root_dir = realfake125k_root_dir

            def create_dataframe(split):
                data = {'filename': [], 'label': []}
                possible_splits = [split, split.capitalize()]
                real_path = None
                fake_path = None
                for s in possible_splits:
                    for real_folder in ['Real', 'real']:
                        for fake_folder in ['Fake', 'fake']:
                            temp_real = os.path.join(root_dir, s, real_folder)
                            temp_fake = os.path.join(root_dir, s, fake_folder)
                            print(f"Checking split: {s}, Real folder: {real_folder}, Fake folder: {fake_folder}")
                            print(f"Real path: {temp_real}, Exists: {os.path.exists(temp_real)}")
                            print(f"Fake path: {temp_fake}, Exists: {os.path.exists(temp_fake)}")
                            if os.path.exists(temp_real) or os.path.exists(temp_fake):
                                real_path = temp_real
                                fake_path = temp_fake
                                break
                        if real_path and fake_path:
                            break
                    if real_path and fake_path:
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
            # If test directory doesn't exist, use validation data as test data
            try:
                test_data = create_dataframe('test')
            except FileNotFoundError:
                print("Warning: Test directory not found, using validation data as test data")
                test_data = val_data.copy()

            train_data = train_data.sample(frac=1, random_state=3407).reset_index(drop=True)
            val_data = val_data.sample(frac=1, random_state=3407).reset_index(drop=True)
            test_data = test_data.sample(frac=1, random_state=3407).reset_index(drop=True)

        # Reset indices
        train_data = train_data.reset_index(drop=True)
        val_data = val_data.reset_index(drop=True)
        test_data = test_data.reset_index(drop=True)

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
        dataset_mode='125k',
        realfake125k_root_dir='/kaggle/input/dfdc-faces-of-the-train-sample',
        train_batch_size=32,
        eval_batch_size=32,
        ddp=False,
    )
