import csv
import os
import torch
from torch.utils.data import DataLoader
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from sklearn.model_selection import train_test_split
import random
from collections import Counter

class FaceDALIPipeline(Pipeline):
    def __init__(self, data_list, root_dir, batch_size, num_threads, device_id, is_training, image_size, mean, std):
        super(FaceDALIPipeline, self).__init__(batch_size, num_threads, device_id, seed=3407)
        self.data_list = data_list  # List of (img_path, label) tuples
        self.root_dir = root_dir
        self.is_training = is_training
        self.image_size = image_size
        self.mean = mean
        self.std = std

        # Define DALI pipeline
        self.input = fn.readers.file(
            files=[os.path.join(root_dir, img_path) for img_path, _ in data_list],
            labels=[int(label in ['real', 'Real', 1]) for _, label in data_list],
            shuffle=is_training,
        )

        # Decode images
        self.decode = fn.decoders.image(device="mixed", output_type=types.RGB)

        # Resize
        self.resize = fn.resize(self.decode, resize_x=image_size[0], resize_y=image_size[1])

        # Training augmentations
        if is_training:
            self.augment = fn.random_resized_crop(
                self.resize,
                size=image_size[0],
                random_area=[0.8, 1.2],
                random_aspect_ratio=[0.9, 1.1]
            )
            self.flip = fn.flip(self.augment, horizontal=1, probability=0.5)
            self.rotate = fn.rotate(self.flip, angle=fn.random.uniform(range=(-20, 20)), probability=0.5)
            self.color = fn.color_twist(
                self.rotate,
                brightness=fn.random.uniform(range=(0.7, 1.3)),
                contrast=fn.random.uniform(range=(0.7, 1.3)),
                saturation=fn.random.uniform(range=(0.7, 1.3)),
                probability=0.5
            )
            self.final = self.color
        else:
            self.final = self.resize

        # Normalize
        self.normalize = fn.crop_mirror_normalize(
            self.final,
            dtype=types.FLOAT,
            output_layout="CHW",
            mean=[m * 255 for m in mean],
            std=[s * 255 for s in std]
        )

    def define_graph(self):
        images, labels = self.input
        images = self.decode(images)
        images = self.normalize
        return images, labels.gpu()

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
        num_workers=4,
        device_id=0,
        ddp=False,
    ):
        if dataset_mode not in ['hardfake', 'rvf10k', '140k']:
            raise ValueError("dataset_mode must be 'hardfake', 'rvf10k', or '140k'")

        self.dataset_mode = dataset_mode
        self.device_id = device_id

        # Define image size and mean/std based on dataset_mode
        image_size = (256, 256) if dataset_mode in ['rvf10k', '140k'] else (300, 300)
        if dataset_mode == 'hardfake':
            mean = (0.5124, 0.4165, 0.3684)
            std = (0.2363, 0.2087, 0.2029)
        elif dataset_mode == 'rvf10k':
            mean = (0.5212, 0.4260, 0.3811)
            std = (0.2486, 0.2238, 0.2211)
        else:  # dataset_mode == '140k'
            mean = (0.5207, 0.4258, 0.3806)
            std = (0.2490, 0.2239, 0.2212)

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

        # Create DALI pipelines
        self.loader_train = self._create_dali_loader(
            train_data, root_dir, train_batch_size, num_workers, device_id, is_training=True,
            image_size=image_size, mean=mean, std=std
        )
        self.loader_val = self._create_dali_loader(
            val_data, root_dir, eval_batch_size, num_workers, device_id, is_training=False,
            image_size=image_size, mean=mean, std=std
        )
        self.loader_test = self._create_dali_loader(
            test_data, root_dir, eval_batch_size, num_workers, device_id, is_training=False,
            image_size=image_size, mean=mean, std=std
        )

        # Debug: Print loader sizes
        print(f"Train loader batches: {len(self.loader_train)}")
        print(f"Validation loader batches: {len(self.loader_val)}")
        print(f"Test loader batches: {len(self.loader_test)}")

    def _create_dali_loader(self, data, root_dir, batch_size, num_threads, device_id, is_training, image_size, mean, std):
        pipeline = FaceDALIPipeline(
            data, root_dir, batch_size, num_threads, device_id, is_training, image_size, mean, std
        )
        return DALIGenericIterator(
            pipeline,
            ["data", "label"],
            size=len(data),
            auto_reset=True
        )


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
