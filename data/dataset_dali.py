import os
import pandas as pd
import torch
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
from sklearn.model_selection import train_test_split

class HybridTrainPipeline(Pipeline):
    def __init__(self, data_frame, root_dir, batch_size, num_threads, device_id, img_column, image_size, mean, std, local_rank=0, world_size=1):
        super().__init__(batch_size, num_threads, device_id, seed=-1)
        self.data_frame = data_frame
        self.root_dir = root_dir
        self.img_column = img_column
        self.image_size = image_size
        self.mean = [m * 255 for m in mean]  # Scale mean for DALI
        self.std = [s * 255 for s in std]    # Scale std for DALI

        # Create a temporary file list for DALI FileReader
        file_list_path = f"file_list_{device_id}.txt"
        with open(file_list_path, 'w') as f:
            for idx in range(len(data_frame)):
                img_path = os.path.join(root_dir, data_frame[self.img_column].iloc[idx])
                label = data_frame['label'].iloc[idx]
                label = 1 if label in [1, 'real', 'Real'] else 0
                f.write(f"{img_path} {label}\n")
        
        self.input = fn.readers.file(file_list=file_list_path, random_shuffle=True, name="Reader")
        self.decode = fn.decoders.image(device="mixed", output_type=types.RGB)
        self.resize = fn.random_resized_crop(
            device="gpu",
            size=self.image_size,
            random_area=[0.8, 1.2],
            random_aspect_ratio=[0.9, 1.1]
        )
        self.cmnp = fn.crop_mirror_normalize(
            device="gpu",
            dtype=types.FLOAT,
            output_layout="CHW",
            crop=self.image_size,
            mean=self.mean,
            std=self.std,
            mirror=fn.random.coin_flip(probability=0.5)
        )
        self.color_jitter = fn.color_twist(
            device="gpu",
            brightness=fn.random.uniform(range=[0.7, 1.3]),
            contrast=fn.random.uniform(range=[0.7, 1.3]),
            saturation=fn.random.uniform(range=[0.7, 1.3])
        )
        self.rotate = fn.rotate(
            device="gpu",
            angle=fn.random.uniform(range=[-20, 20]),
            interp_type=types.INTERP_LINEAR
        )

    def define_graph(self):
        images, labels = self.input
        images = self.decode(images)
        images = self.resize(images)
        images = self.rotate(images)
        images = self.color_jitter(images)
        images = self.cmnp(images)
        return images, labels

class HybridValPipeline(Pipeline):
    def __init__(self, data_frame, root_dir, batch_size, num_threads, device_id, img_column, image_size, mean, std):
        super().__init__(batch_size, num_threads, device_id, seed=-1)
        self.data_frame = data_frame
        self.root_dir = root_dir
        self.img_column = img_column
        self.image_size = image_size
        self.mean = [m * 255 for m in mean]
        self.std = [s * 255 for s in std]

        file_list_path = f"file_list_val_{device_id}.txt"
        with open(file_list_path, 'w') as f:
            for idx in range(len(data_frame)):
                img_path = os.path.join(root_dir, data_frame[self.img_column].iloc[idx])
                label = data_frame['label'].iloc[idx]
                label = 1 if label in [1, 'real', 'Real'] else 0
                f.write(f"{img_path} {label}\n")
        
        self.input = fn.readers.file(file_list=file_list_path, random_shuffle=False, name="Reader")
        self.decode = fn.decoders.image(device="mixed", output_type=types.RGB)
        self.resize = fn.resize(
            device="gpu",
            resize_shorter=max(self.image_size) * 1.125,
            interp_type=types.INTERP_TRIANGULAR
        )
        self.cmnp = fn.crop_mirror_normalize(
            device="gpu",
            dtype=types.FLOAT,
            output_layout="CHW",
            crop=self.image_size,
            mean=self.mean,
            std=self.std
        )

    def define_graph(self):
        images, labels = self.input
        images = self.decode(images)
        images = self.resize(images)
        images = self.cmnp(images)
        return images, labels

class DatasetSelectorDALI:
    def __init__(
        self,
        dataset_mode,
        hardfake_csv_file=None,
        hardfake_root_dir=None,
        rvf10k_train_csv=None,
        rvf10k_valid_csv=None,
        rvf10k_root_dir=None,
        realfake140k_train_csv=None,
        realfake140k_valid_csv=None,
        realfake140k_test_csv=None,
        realfake140k_root_dir=None,
        realfake200k_train_csv=None,
        realfake200k_val_csv=None,
        realfake200k_test_csv=None,
        realfake200k_root_dir=None,
        train_batch_size=32,
        eval_batch_size=32,
        num_threads=8,
        device_id=0,
        ddp=False,
        local_rank=0,
        world_size=1
    ):
        if dataset_mode not in ['hardfake', 'rvf10k', '140k', '200k']:
            raise ValueError("dataset_mode must be 'hardfake', 'rvf10k', '140k', or '200k'")

        self.dataset_mode = dataset_mode
        self.image_size = (256, 256) if dataset_mode in ['rvf10k', '140k', '200k'] else (300, 300)

        if dataset_mode == 'hardfake':
            mean = (0.5124, 0.4165, 0.3684)
            std = (0.2363, 0.2087, 0.2029)
        elif dataset_mode == 'rvf10k':
            mean = (0.5212, 0.4260, 0.3811)
            std = (0.2486, 0.2238, 0.2211)
        elif dataset_mode == '140k':
            mean = (0.5207, 0.4258, 0.3806)
            std = (0.2490, 0.2239, 0.2212)
        else:  # 200k
            mean = (0.4868, 0.3972, 0.3624)
            std = (0.2296, 0.2066, 0.2009)

        img_column = 'filename' if dataset_mode in ['140k', '200k'] else 'images_id'

        # Load data based on dataset_mode
        if dataset_mode == 'hardfake':
            if not hardfake_csv_file or not hardfake_root_dir:
                raise ValueError("hardfake_csv_file and hardfake_root_dir must be provided")
            full_data = pd.read_csv(hardfake_csv_file)

            def create_image_path(row):
                folder = 'fake' if row['label'] == 'fake' else 'real'
                img_name = row['images_id']
                img_name = os.path.basename(img_name)
                if not img_name.endswith('.jpg'):
                    img_name += '.jpg'
                return os.path.join(folder, img_name)

            full_data['images_id'] = full_data.apply(create_image_path, axis=1)
            root_dir = hardfake_root_dir

            train_data, temp_data = train_test_split(
                full_data, test_size=0.3, stratify=full_data['label'], random_state=3407
            )
            val_data, test_data = train_test_split(
                temp_data, test_size=0.5, stratify=temp_data['label'], random_state=3407
            )
            train_data = train_data.reset_index(drop=True)
            val_data = val_data.reset_index(drop=True)
            test_data = test_data.reset_index(drop=True)

        elif dataset_mode == 'rvf10k':
            if not rvf10k_train_csv or not rvf10k_valid_csv or not rvf10k_root_dir:
                raise ValueError("rvf10k_train_csv, rvf10k_valid_csv, and rvf10k_root_dir must be provided")
            train_data = pd.read_csv(rvf10k_train_csv)

            def create_image_path(row, split='train'):
                folder = 'fake' if row['label'] == 0 else 'real'
                img_name = row['id']
                img_name = os.path.basename(img_name)
                if not img_name.endswith('.jpg'):
                    img_name += '.jpg'
                return os.path.join('rvf10k', split, folder, img_name)

            train_data['images_id'] = train_data.apply(lambda row: create_image_path(row, 'train'), axis=1)
            valid_data = pd.read_csv(rvf10k_valid_csv)
            valid_data['images_id'] = valid_data.apply(lambda row: create_image_path(row, 'valid'), axis=1)

            val_data, test_data = train_test_split(
                valid_data, test_size=0.5, stratify=valid_data['label'], random_state=3407
            )
            val_data = val_data.reset_index(drop=True)
            test_data = test_data.reset_index(drop=True)
            root_dir = rvf10k_root_dir

        elif dataset_mode == '140k':
            if not realfake140k_train_csv or not realfake140k_valid_csv or not realfake140k_test_csv or not realfake140k_root_dir:
                raise ValueError("realfake140k_train_csv, realfake140k_valid_csv, realfake140k_test_csv, and realfake140k_root_dir must be provided")
            train_data = pd.read_csv(realfake140k_train_csv)
            val_data = pd.read_csv(realfake140k_valid_csv)
            test_data = pd.read_csv(realfake140k_test_csv)
            root_dir = os.path.join(realfake140k_root_dir, 'real_vs_fake', 'real-vs-fake')

            if 'filename' not in train_data.columns and 'image' not in train_data.columns and 'path' not in train_data.columns:
                raise ValueError("CSV files for 140k dataset must contain a 'filename', 'image', or 'path' column")

            train_data = train_data.sample(frac=1, random_state=3407).reset_index(drop=True)
            val_data = personally identifiable information(val_data, frac=1, random_state=3407).reset_index(drop=True)
            test_data = test_data.sample(frac=1, random_state=3407).reset_index(drop=True)

        else:  # dataset_mode == '200k'
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

            train_data = train_data.sample(frac=1, random_state=3407).reset_index(drop=True)
            val_data = val_data.sample(frac=1, random_state=3407).reset_index(drop=True)
            test_data = test_data.sample(frac=1, random_state=3407).reset_index(drop=True)

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

        # Create DALI pipelines
        self.train_pipeline = HybridTrainPipeline(
            train_data, root_dir, train_batch_size, num_threads, device_id, img_column, self.image_size, mean, std, local_rank, world_size
        )
        self.val_pipeline = HybridValPipeline(
            val_data, root_dir, eval_batch_size, num_threads, device_id, img_column, self.image_size, mean, std
        )
        self.test_pipeline = HybridValPipeline(
            test_data, root_dir, eval_batch_size, num_threads, device_id, img_column, self.image_size, mean, std
        )

        # Build pipelines
        self.train_pipeline.build()
        self.val_pipeline.build()
        self.test_pipeline.build()

        # Create DALI iterators
        self.loader_train = DALIClassificationIterator(
            self.train_pipeline,
            size=len(train_data) // world_size if ddp else len(train_data),
            auto_reset=True
        )
        self.loader_val = DALIClassificationIterator(
            self.val_pipeline,
            size=len(val_data),
            auto_reset=True
        )
        self.loader_test = DALIClassificationIterator(
            self.test_pipeline,
            size=len(test_data),
            auto_reset=True
        )

        print(f"Train loader batches: {len(self.loader_train)}")
        print(f"Validation loader batches: {len(self.loader_val)}")
        print(f"Test loader batches: {len(self.loader_test)}")

if __name__ == "__main__":
    dataset_hardfake = DatasetSelectorDALI(
        dataset_mode='hardfake',
        hardfake_csv_file='/kaggle/input/hardfakevsrealfaces/data.csv',
        hardfake_root_dir='/kaggle/input/hardfakevsrealfaces',
        train_batch_size=256,
        eval_batch_size=128,
        num_threads=8,
        device_id=0,
        ddp=True,
    )

    dataset_rvf10k = DatasetSelectorDALI(
        dataset_mode='rvf10k',
        rvf10k_train_csv='/kaggle/input/rvf10k/train.csv',
        rvf10k_valid_csv='/kaggle/input/rvf10k/valid.csv',
        rvf10k_root_dir='/kaggle/input/rvf10k',
        train_batch_size=256,
        eval_batch_size=128,
        num_threads=8,
        device_id=0,
        ddp=True,
    )

    dataset_140k = DatasetSelectorDALI(
        dataset_mode='140k',
        realfake140k_train_csv='/kaggle/input/140k-real-and-fake-faces/train.csv',
        realfake140k_valid_csv='/kaggle/input/140k-real-and-fake-faces/valid.csv',
        realfake140k_test_csv='/kaggle/input/140k-real-and-fake-faces/test.csv',
        realfake140k_root_dir='/kaggle/input/140k-real-and-fake-faces',
        train_batch_size=256,
        eval_batch_size=128,
        num_threads=8,
        device_id=0,
        ddp=True,
    )

    dataset_200k = DatasetSelectorDALI(
        dataset_mode='200k',
        realfake200k_train_csv='/kaggle/input/200k-real-vs-ai-visuals-by-mbilal/train_labels.csv',
        realfake200k_val_csv='/kaggle/input/200k-real-vs-ai-visuals-by-mbilal/val_labels.csv',
        realfake200k_test_csv='/kaggle/input/200k-real-vs-ai-visuals-by-mbilal/test_labels.csv',
        realfake200k_root_dir='/kaggle/input/200k-real-vs-ai-visuals-by-mbilal/my_real_vs_ai_dataset',
        train_batch_size=128,
        eval_batch_size=128,
        num_threads=8,
        device_id=0,
        ddp=True,
    )
