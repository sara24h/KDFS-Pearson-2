import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torch.amp import autocast, GradScaler
from PIL import Image
import argparse
import random
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image as IPImage, display
from ptflops import get_model_complexity_info
from data.dataset import FaceDataset, Dataset_selector

def parse_args():
    parser = argparse.ArgumentParser(description='Train a ResNet50 model with single output for fake vs real face classification.')
    parser.add_argument('--dataset_mode', type=str, required=True, choices=['hardfake', 'rvf10k', '140k', '200k', '125k'],
                        help='Dataset to use: hardfake, rvf10k, 140k, 200k, or 125k')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the dataset directory containing images and CSV file(s)')
    parser.add_argument('--teacher_dir', type=str, default='teacher_dir',
                        help='Directory to save the trained model and outputs')
    parser.add_argument('--img_height', type=int, default=300,
                        help='Height of input images (default: 300 for hardfake, 256 for rvf10k, 140k, 200k, 160 for 125k)')
    parser.add_argument('--img_width', type=int, default=300,
                        help='Width of input images (default: 300 for hardfake, 256 for rvf10k, 140k, 200k, 160 for 125k)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate for the optimizer')
    return parser.parse_args()

args = parse_args()

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

dataset_mode = args.dataset_mode
data_dir = args.data_dir
teacher_dir = args.teacher_dir
img_height = 160 if dataset_mode == '125k' else 256 if dataset_mode in ['rvf10k', '140k', '200k'] else args.img_height
img_width = 160 if dataset_mode == '125k' else 256 if dataset_mode in ['rvf10k', '140k', '200k'] else args.img_width
batch_size = args.batch_size
epochs = args.epochs
lr = args.lr
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# بررسی وجود پوشه‌ها
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Directory {data_dir} not found!")
if not os.path.exists(teacher_dir):
    os.makedirs(teacher_dir)

# تابع برای ایجاد DataFrame از ساختار پوشه‌ها
def create_dataframe_from_dir(root_dir, subdirs=['fake', 'real']):
    data = []
    for subdir in subdirs:
        subdir_path = os.path.join(root_dir, subdir)
        if not os.path.exists(subdir_path):
            continue
        label = 1 if subdir == 'real' else 0
        for img_name in os.listdir(subdir_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                rel_path = os.path.join(subdir, img_name)
                data.append({'path': rel_path, 'label': label})
    return pd.DataFrame(data)


# انتخاب دیتاست
if dataset_mode == 'hardfake':
    csv_file = os.path.join(data_dir, 'data.csv')
    if os.path.exists(csv_file):
        dataset = Dataset_selector(
            dataset_mode='hardfake',
            hardfake_csv_file=csv_file,
            hardfake_root_dir=data_dir,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
            ddp=False
        )
    else:
        # ایجاد DataFrame و ذخیره به CSV موقت
        train_df = create_dataframe_from_dir(data_dir)
        temp_csv = os.path.join(teacher_dir, 'temp_hardfake.csv')
        train_df.to_csv(temp_csv, index=False)
        dataset = Dataset_selector(
            dataset_mode='hardfake',
            hardfake_csv_file=temp_csv,
            hardfake_root_dir=data_dir,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
            ddp=False
        )
        print("Hardfake DataFrame:\n", train_df.head())

elif dataset_mode == 'rvf10k':
    train_csv = os.path.join(data_dir, 'train.csv')
    valid_csv = os.path.join(data_dir, 'valid.csv')
    if os.path.exists(train_csv) and os.path.exists(valid_csv):
        dataset = Dataset_selector(
            dataset_mode='rvf10k',
            rvf10k_train_csv=train_csv,
            rvf10k_valid_csv=valid_csv,
            rvf10k_root_dir=data_dir,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
            ddp=False
        )
    else:
        # ایجاد DataFrame و ذخیره به CSV موقت
        train_df = create_dataframe_from_dir(os.path.join(data_dir, 'train'))
        valid_df = create_dataframe_from_dir(os.path.join(data_dir, 'valid'))
        temp_train_csv = os.path.join(teacher_dir, 'temp_train.csv')
        temp_valid_csv = os.path.join(teacher_dir, 'temp_valid.csv')
        train_df.to_csv(temp_train_csv, index=False)
        valid_df.to_csv(temp_valid_csv, index=False)
        dataset = Dataset_selector(
            dataset_mode='rvf10k',
            rvf10k_train_csv=temp_train_csv,
            rvf10k_valid_csv=temp_valid_csv,
            rvf10k_root_dir=data_dir,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
            ddp=False
        )
        print("RVF10k Train DataFrame:\n", train_df.head())
        print("RVF10k Valid DataFrame:\n", valid_df.head())

elif dataset_mode == '140k':
    train_csv = os.path.join(data_dir, 'train.csv')
    valid_csv = os.path.join(data_dir, 'valid.csv')
    test_csv = os.path.join(data_dir, 'test.csv')
    if os.path.exists(train_csv) and os.path.exists(valid_csv):
        dataset = Dataset_selector(
            dataset_mode='140k',
            realfake140k_train_csv=train_csv,
            realfake140k_valid_csv=valid_csv,
            realfake140k_test_csv=test_csv if os.path.exists(test_csv) else valid_csv,
            realfake140k_root_dir=data_dir,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
            ddp=False
        )
    else:
        # ایجاد DataFrame و ذخیره به CSV موقت
        train_df = create_dataframe_from_dir(os.path.join(data_dir, 'train'))
        valid_df = create_dataframe_from_dir(os.path.join(data_dir, 'valid'))
        test_df = create_dataframe_from_dir(os.path.join(data_dir, 'test')) if os.path.exists(os.path.join(data_dir, 'test')) else valid_df
        temp_train_csv = os.path.join(teacher_dir, 'temp_train.csv')
        temp_valid_csv = os.path.join(teacher_dir, 'temp_valid.csv')
        temp_test_csv = os.path.join(teacher_dir, 'temp_test.csv')
        train_df.to_csv(temp_train_csv, index=False)
        valid_df.to_csv(temp_valid_csv, index=False)
        test_df.to_csv(temp_test_csv, index=False)
        dataset = Dataset_selector(
            dataset_mode='140k',
            realfake140k_train_csv=temp_train_csv,
            realfake140k_valid_csv=temp_valid_csv,
            realfake140k_test_csv=temp_test_csv,
            realfake140k_root_dir=data_dir,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
            ddp=False
        )
        print("140k Train DataFrame:\n", train_df.head())
        print("140k Valid DataFrame:\n", valid_df.head())
        print("140k Test DataFrame:\n", test_df.head())

elif dataset_mode == '200k':
    train_csv = "/kaggle/input/200k-real-vs-ai-visuals-by-mbilal/train_labels.csv"
    valid_csv = "/kaggle/input/200k-real-vs-ai-visuals-by-mbilal/val_labels.csv"
    test_csv = "/kaggle/input/200k-real-vs-ai-visuals-by-mbilal/test_labels.csv"
    if os.path.exists(train_csv) and os.path.exists(valid_csv):
        dataset = Dataset_selector(
            dataset_mode='200k',
            realfake200k_train_csv=train_csv,
            realfake200k_val_csv=valid_csv,
            realfake200k_test_csv=test_csv if os.path.exists(test_csv) else valid_csv,
            realfake200k_root_dir=data_dir,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
            ddp=False
        )
    else:
        # ایجاد DataFrame و ذخیره به CSV موقت
        train_df = create_dataframe_from_dir(os.path.join(data_dir, 'train'))
        valid_df = create_dataframe_from_dir(os.path.join(data_dir, 'valid'))
        test_df = create_dataframe_from_dir(os.path.join(data_dir, 'test')) if os.path.exists(os.path.join(data_dir, 'test')) else valid_df
        temp_train_csv = os.path.join(teacher_dir, 'temp_train.csv')
        temp_valid_csv = os.path.join(teacher_dir, 'temp_valid.csv')
        temp_test_csv = os.path.join(teacher_dir, 'temp_test.csv')
        train_df.to_csv(temp_train_csv, index=False)
        valid_df.to_csv(temp_valid_csv, index=False)
        test_df.to_csv(temp_test_csv, index=False)
        dataset = Dataset_selector(
            dataset_mode='200k',
            realfake200k_train_csv=temp_train_csv,
            realfake200k_val_csv=temp_valid_csv,
            realfake200k_test_csv=temp_test_csv,
            realfake200k_root_dir=data_dir,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
            ddp=False
        )
        print("200k Train DataFrame:\n", train_df.head())
        print("200k Valid DataFrame:\n", valid_df.head())
        print("200k Test DataFrame:\n", test_df.head())

elif dataset_mode == '125k':
    # برای 125k از FaceDataset مستقیماً استفاده می‌کنیم
    train_df = create_dataframe_from_dir(os.path.join(data_dir, 'train'))
    valid_df = create_dataframe_from_dir(os.path.join(data_dir, 'validation'))
    test_df = valid_df  # استفاده از validation به عنوان تست

    # چاپ برای دیباگ
    print("125k Train DataFrame:\n", train_df.head())
    print("125k Validation DataFrame:\n", valid_df.head())
    print("Root directory:", data_dir)

    # ایجاد دیتاست‌ها
    train_dataset = FaceDataset(train_df, data_dir, transform=transform)
    val_dataset = FaceDataset(valid_df, data_dir, transform=transform)
    test_dataset = FaceDataset(test_df, data_dir, transform=transform)

    # ایجاد لودرها
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
else:
    raise ValueError("Invalid dataset_mode. Choose 'hardfake', 'rvf10k', '140k', '200k', or '125k'.")

# اگر از Dataset_selector استفاده شده، لودرها را از آن بگیریم
if dataset_mode != '125k':
    train_loader = dataset.loader_train
    val_loader = dataset.loader_val
    test_loader = dataset.loader_test

# چاپ تعداد داده‌ها برای دیباگ
print("Train loader size:", len(train_loader))
print("Validation loader size:", len(val_loader))
print("Test loader size:", len(test_loader))

# تعریف مدل
model = models.resnet50(weights='IMAGENET1K_V1')
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)
model = model.to(device)

# فریز کردن لایه‌ها به جز layer4 و fc
for param in model.parameters():
    param.requires_grad = False
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-5},
    {'params': model.fc.parameters(), 'lr': lr}
], weight_decay=1e-4)

scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

if device.type == 'cuda':
    torch.cuda.empty_cache()

best_val_acc = 0.0
best_model_path = os.path.join(teacher_dir, 'teacher_model_best.pth')

# حلقه آموزش
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

    # اعتبارسنجی
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

# ذخیره مدل نهایی
torch.save(model.state_dict(), os.path.join(teacher_dir, 'teacher_model_final.pth'))
print(f'Saved final model at epoch {epochs}')

# تست مدل
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

# آماده‌سازی داده‌ها برای visualization
val_data = test_loader.dataset.data
transform_test = test_loader.dataset.transform

# تنظیم ستون تصویر
if dataset_mode == '140k':
    img_column = 'path'
elif dataset_mode == '200k':
    img_column = 'filename'
elif dataset_mode == '125k':
    img_column = 'path'
else:
    img_column = 'images_id'

if img_column not in val_data.columns:
    print(f"Warning: Column '{img_column}' not found in DataFrame. Available columns: {list(val_data.columns)}")
    img_column = 'path'  # بازگشت به 'path' اگر ستون پیش‌فرض موجود نباشد

# انتخاب نمونه‌های تصادفی برای نمایش
random_indices = random.sample(range(len(val_data)), min(10, len(val_data)))
fig, axes = plt.subplots(2, 5, figsize=(15, 8))
axes = axes.ravel()

with torch.no_grad():
    for i, idx in enumerate(random_indices):
        row = val_data.iloc[idx]
        img_name = row[img_column]
        label = row['label']
        
        # ساخت مسیر کامل تصویر
        if dataset_mode == '200k':
            subfolder = 'real' if label == 1 else 'ai_images'
            img_path = os.path.join(data_dir, 'my_real_vs_ai_dataset', 'my_real_vs_ai_dataset', subfolder, img_name)
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

# محاسبه پیچیدگی مدل
for param in model.parameters():
    param.requires_grad = True

flops, params = get_model_complexity_info(model, (3, img_height, img_width), as_strings=True, print_per_layer_stat=True)
print('FLOPs:', flops)
print('Parameters:', params)
