import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms, models, datasets
from torch.amp import autocast, GradScaler
from PIL import Image
import argparse
import random
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image as IPImage, display
from ptflops import get_model_complexity_info

# تعریف Dataset_selector
class Dataset_selector:
    def __init__(self, dataset_mode, train_batch_size, eval_batch_size, num_workers, pin_memory, ddp, **kwargs):
        self.dataset_mode = dataset_mode
        self.transform = transforms.Compose([
            transforms.Resize((160, 160) if dataset_mode == '125k' else (256, 256) if dataset_mode in ['rvf10k', '140k', '200k'] else (300, 300)),
            transforms.ToTensor()
        ])
        
        if dataset_mode == 'hardfake':
            csv_file = kwargs.get('hardfake_csv_file')
            root_dir = kwargs.get('hardfake_root_dir')
            data = pd.read_csv(csv_file)
            self.loader_train = DataLoader(
                FaceDataset(data, root_dir, self.transform, split='train'),
                batch_size=train_batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
            )
            self.loader_val = DataLoader(
                FaceDataset(data, root_dir, self.transform, split='val'),
                batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
            )
            self.loader_test = DataLoader(
                FaceDataset(data, root_dir, self.transform, split='test'),
                batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
            )
        elif dataset_mode in ['rvf10k', '140k', '200k']:
            train_csv = kwargs.get(f'realfake{dataset_mode}_train_csv')
            val_csv = kwargs.get(f'realfake{dataset_mode}_val_csv')
            test_csv = kwargs.get(f'realfake{dataset_mode}_test_csv')
            root_dir = kwargs.get(f'realfake{dataset_mode}_root_dir')
            train_data = pd.read_csv(train_csv)
            val_data = pd.read_csv(val_csv)
            test_data = pd.read_csv(test_csv) if test_csv else val_data
            self.loader_train = DataLoader(
                FaceDataset(train_data, root_dir, self.transform, split='train'),
                batch_size=train_batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
            )
            self.loader_val = DataLoader(
                FaceDataset(val_data, root_dir, self.transform, split='val'),
                batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
            )
            self.loader_test = DataLoader(
                FaceDataset(test_data, root_dir, self.transform, split='test'),
                batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
            )
        elif dataset_mode == '125k':
            train_dataset = datasets.ImageFolder(os.path.join(kwargs.get('realfake125k_root_dir'), 'train'), transform=self.transform)
            val_dataset = datasets.ImageFolder(os.path.join(kwargs.get('realfake125k_root_dir'), 'validation'), transform=self.transform)
            
            # تقسیم دیتاست validation به دو بخش: نصف برای validation و نصف برای تست
            val_len = len(val_dataset)
            val_subset_len = val_len // 2
            test_subset_len = val_len - val_subset_len
            val_subset, test_subset = random_split(val_dataset, [val_subset_len, test_subset_len])
            
            self.loader_train = DataLoader(
                train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
            )
            self.loader_val = DataLoader(
                val_subset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
            )
            self.loader_test = DataLoader(
                test_subset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
            )
            # تنظیم DataFrame برای تست
            self.loader_test.dataset.data = pd.DataFrame({
                'filename': [os.path.join(*val_dataset.imgs[i][0].split(os.sep)[-3:]) for i in test_subset.indices],
                'label': [val_dataset.imgs[i][1] for i in test_subset.indices]
            })
            # تنظیم DataFrame برای validation
            self.loader_val.dataset.data = pd.DataFrame({
                'filename': [os.path.join(*val_dataset.imgs[i][0].split(os.sep)[-3:]) for i in val_subset.indices],
                'label': [val_dataset.imgs[i][1] for i in val_subset.indices]
            })

        # چاپ تعداد داده‌ها برای بررسی
        print(f"Number of train samples for {dataset_mode}: {len(self.loader_train.dataset)}")
        print(f"Number of validation samples for {dataset_mode}: {len(self.loader_val.dataset)}")
        print(f"Number of test samples for {dataset_mode}: {len(self.loader_test.dataset)}")

# تعریف FaceDataset برای دیتاست‌های CSV
class FaceDataset:
    def __init__(self, data, root_dir, transform=None, split='train'):
        self.data = data
        self.root_dir = root_dir
        self.transform = transform
        self.split = split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['images_id' if self.split == 'hardfake' else 'filename']
        label = self.data.iloc[idx]['label']
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def parse_args():
    parser = argparse.ArgumentParser(description='Train a ResNet50 model with single output for fake vs real face classification.')
    parser.add_argument('--dataset_mode', type=str, required=True, choices=['hardfake', 'rvf10k', '140k', '200k', '125k'],
                        help='Dataset to use: hardfake, rvf10k, 140k, 200k, or 125k')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the dataset directory containing images and CSV file(s)')
    parser.add_argument('--teacher_dir', type=str, default='teacher_dir',
                        help='Directory to save the trained model and outputs')
    parser.add_argument('--img_height', type=int, default=160,
                        help='Height of input images (default: 160 for 125k, 300 for hardfake, 256 for rvf10k, 140k, 200k)')
    parser.add_argument('--img_width', type=int, default=160,
                        help='Width of input images (default: 160 for 125k, 300 for hardfake, 256 for rvf10k, 140k, 200k)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of training epochs')
    parser.add_argument('--lr_fc', type=float, default=0.0001,
                        help='Learning rate for the fully connected (fc) layer')
    parser.add_argument('--lr_layer4', type=float, default=1e-5,
                        help='Learning rate for the layer4 of ResNet50')
    return parser.parse_args()

args = parse_args()

# اعتبارسنجی نرخ‌های یادگیری
if args.lr_fc <= 0 or args.lr_layer4 <= 0:
    raise ValueError("Learning rates must be positive!")

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

dataset_mode = args.dataset_mode
data_dir = args.data_dir
teacher_dir = args.teacher_dir
img_height = 160 if dataset_mode == '125k' else (256 if dataset_mode in ['rvf10k', '140k', '200k'] else args.img_height)
img_width = 160 if dataset_mode == '125k' else (256 if dataset_mode in ['rvf10k', '140k', '200k'] else args.img_width)
batch_size = args.batch_size
epochs = args.epochs
lr_fc = args.lr_fc
lr_layer4 = args.lr_layer4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Directory {data_dir} not found!")
if not os.path.exists(teacher_dir):
    os.makedirs(teacher_dir)

if dataset_mode == 'hardfake':
    dataset = Dataset_selector(
        dataset_mode='hardfake',
        hardfake_csv_file=os.path.join(data_dir, 'data.csv'),
        hardfake_root_dir=data_dir,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        ddp=False
    )
elif dataset_mode == 'rvf10k':
    dataset = Dataset_selector(
        dataset_mode='rvf10k',
        realfake10k_train_csv=os.path.join(data_dir, 'train.csv'),
        realfake10k_valid_csv=os.path.join(data_dir, 'valid.csv'),
        realfake10k_root_dir=data_dir,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        ddp=False
    )
elif dataset_mode == '140k':
    dataset = Dataset_selector(
        dataset_mode='140k',
        realfake140k_train_csv=os.path.join(data_dir, 'train.csv'),
        realfake140k_valid_csv=os.path.join(data_dir, 'valid.csv'),
        realfake140k_test_csv=os.path.join(data_dir, 'test.csv'),
        realfake140k_root_dir=data_dir,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        ddp=False
    )
elif dataset_mode == '200k':
    dataset = Dataset_selector(
        dataset_mode='200k',
        realfake200k_train_csv="/kaggle/input/200k-real-vs-ai-visuals-by-mbilal/train_labels.csv",
        realfake200k_val_csv="/kaggle/input/200k-real-vs-ai-visuals-by-mbilal/val_labels.csv",
        realfake200k_test_csv="/kaggle/input/200k-real-vs-ai-visuals-by-mbilal/test_labels.csv",
        realfake200k_root_dir=data_dir,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        ddp=False
    )
elif dataset_mode == '125k':
    dataset = Dataset_selector(
        dataset_mode='125k',
        realfake125k_root_dir=data_dir,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        ddp=False
    )
else:
    raise ValueError("Invalid dataset_mode. Choose 'hardfake', 'rvf10k', '140k', '200k', or '125k'.")

train_loader = dataset.loader_train
val_loader = dataset.loader_val
test_loader = dataset.loader_test

model = models.resnet50(weights='IMAGENET1K_V1')
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)
model = model.to(device)

for param in model.parameters():
    param.requires_grad = False
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam([
    {'params': model.layer4.parameters(), 'lr': lr_layer4},
    {'params': model.fc.parameters(), 'lr': lr_fc}
], weight_decay=1e-4)

scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

if device.type == 'cuda':
    torch.cuda.empty_cache()

best_val_acc = 0.0
best_model_path = os.path.join(teacher_dir, 'teacher_model_best.pth')

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device).float()
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)

        if device.type == 'cuda':
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct_train += (preds == labels).sum().item()
        total_train += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct_train / total_train
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')

    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device).float()
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                outputs = model(images).squeeze(1)
                loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct_val += (preds == labels).sum().item()
            total_val += labels.size(0)

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct_val / total_val
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        torch.save(model.state_dict(), best_model_path)
        print(f'Saved best model with validation accuracy: {val_accuracy:.2f}% at epoch {epoch+1}')

torch.save(model.state_dict(), os.path.join(teacher_dir, 'teacher_model_final.pth'))
print(f'Saved final model at epoch {epochs}')

model.eval()
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device).float()
        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)
        test_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
print(f'Test Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {100 * correct / total:.2f}%')

val_data = dataset.loader_test.dataset.data
transform_test = dataset.loader_test.dataset.transform

if dataset_mode == '140k':
    img_column = 'path'
elif dataset_mode in ['200k', '125k']:
    img_column = 'filename'
else:
    img_column = 'images_id'

if img_column not in val_data.columns and dataset_mode != '125k':
    raise KeyError(f"Column '{img_column}' not found in DataFrame. Available columns: {list(val_data.columns)}")

random_indices = random.sample(range(len(val_data)), min(10, len(val_data)))
fig, axes = plt.subplots(2, 5, figsize=(15, 8))
axes = axes.ravel()

with torch.no_grad():
    for i, idx in enumerate(random_indices):
        row = val_data.iloc[idx]
        img_name = row[img_column]
        label = row['label']
        
        if dataset_mode == '140k':
            img_path = os.path.join(data_dir, img_name)
        elif dataset_mode in ['200k', '125k']:
            subfolder = 'real' if label == 1 else 'fake'
            img_path = os.path.join(data_dir, 'train' if 'train' in img_name else 'validation', subfolder, os.path.basename(img_name))
        else:
            img_path = os.path.join(data_dir, img_name)

        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            axes[i].set_title("Image not found")
            axes[i].axis('off')
            continue
        image = Image.open(img_path).convert('RGB')
        image_transformed = transform_test(image).unsqueeze(0).to(device)
        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            output = model(image_transformed).squeeze(1)
        prob = torch.sigmoid(output).item()
        predicted_label = 'real' if prob > 0.5 else 'fake'
        true_label = 'real' if label == 1 else 'fake'
        axes[i].imshow(image)
        axes[i].set_title(f'True: {true_label}\nPred: {predicted_label}', fontsize=10)
        axes[i].axis('off')
        print(f"Image: {img_path}, True Label: {true_label}, Predicted: {predicted_label}")

plt.tight_layout()
file_path = os.path.join(teacher_dir, 'test_samples.png')
plt.savefig(file_path)
display(IPImage(filename=file_path))

for param in model.parameters():
    param.requires_grad = True

flops, params = get_model_complexity_info(model, (3, img_height, img_width), as_strings=True, print_per_layer_stat=True)
print('FLOPs:', flops)
print('Parameters:', params)
