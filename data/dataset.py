import json
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import utils, loss, meter, scheduler
from data.dataset import Dataset_selector, FaceDataset
from model.student.ResNet_sparse import ResNet_50_sparse_hardfakevsreal
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data.distributed import DistributedSampler
import glob


class FinetuneDDP:
    def __init__(self, args):
        """Initialize FinetuneDDP with provided arguments."""
        self.args = args
        self.dataset_dir = args.dataset_dir
        self.dataset_mode = args.dataset_mode
        if self.dataset_mode == "hardfake":
            self.dataset_type = "hardfakevsrealfaces"
        elif self.dataset_mode == "rvf10k":
            self.dataset_type = "rvf10k"
        elif self.dataset_mode == "140k":
            self.dataset_type = "140k"
        elif self.dataset_mode == "200k":
            self.dataset_type = "200k"
        elif self.dataset_mode == "190k":
            self.dataset_type = "190k"
        elif self.dataset_mode == "330k":
            self.dataset_type = "330k"
        else:
            raise ValueError(f"Unknown dataset_mode: {self.dataset_mode}")
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.arch = args.arch
        self.seed = args.seed
        self.result_dir = args.result_dir
        self.finetune_train_batch_size = args.finetune_train_batch_size
        self.finetune_eval_batch_size = args.finetune_eval_batch_size
        self.finetune_student_ckpt_path = args.finetune_student_ckpt_path
        self.finetune_num_epochs = args.finetune_num_epochs
        self.finetune_lr = args.finetune_lr
        self.finetune_warmup_steps = args.finetune_warmup_steps
        self.finetune_warmup_start_lr = args.finetune_warmup_start_lr
        self.finetune_lr_decay_T_max = args.finetune_lr_decay_T_max
        self.finetune_lr_decay_eta_min = args.finetune_lr_decay_eta_min
        self.finetune_weight_decay = args.finetune_weight_decay
        self.finetune_resume = args.finetune_resume
        self.sparsed_student_ckpt_path = args.sparsed_student_ckpt_path

        self.start_epoch = 0
        self.best_prec1_after_finetune = 0
        self.world_size = 0
        self.local_rank = -1
        self.rank = -1

    def dist_init(self):
        """Initialize distributed training with NCCL backend."""
        os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"
        dist.init_process_group("nccl")
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(self.local_rank)

    def result_init(self):
        """Initialize logging and TensorBoard writer for rank 0."""
        if self.rank == 0:
            self.writer = SummaryWriter(self.result_dir)
            self.logger = utils.get_logger(
                os.path.join(self.result_dir, "finetune_logger.log"), "finetune_logger"
            )
            self.logger.info("Finetune configuration:")
            self.logger.info(str(json.dumps(vars(self.args), indent=4)))
            utils.record_config(
                self.args, os.path.join(self.result_dir, "finetune_config.txt")
            )
            self.logger.info("--------- Finetune -----------")

    def setup_seed(self):
        """Set random seeds for reproducibility."""
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        torch.use_deterministic_algorithms(True)
        self.seed += self.rank
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.enabled = True

    def dataload(self):
        """Load dataset based on dataset_mode."""
        # Common arguments for Dataset_selector
        dataset_args = {
            'dataset_mode': self.dataset_mode,
            'train_batch_size': self.finetune_train_batch_size,
            'eval_batch_size': self.finetune_eval_batch_size,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'ddp': True,
        }

        # Update dataset_args based on dataset_mode
        if self.dataset_mode == 'hardfake':
            hardfake_csv_file = self.args.hardfake_csv_file
            if not os.path.exists(hardfake_csv_file):
                raise FileNotFoundError(f"Hardfake CSV file not found: {hardfake_csv_file}")
            dataset_args.update({
                'hardfake_csv_file': hardfake_csv_file,
                'hardfake_root_dir': self.dataset_dir
            })
        elif self.dataset_mode == 'rvf10k':
            train_csv = self.args.rvf10k_train_csv
            valid_csv = self.args.rvf10k_valid_csv
            test_csv = self.args.rvf10k_test_csv if hasattr(self.args, 'rvf10k_test_csv') else None
            if not os.path.exists(train_csv):
                raise FileNotFoundError(f"RVF10K train CSV not found: {train_csv}")
            if not os.path.exists(valid_csv):
                raise FileNotFoundError(f"RVF10K valid CSV not found: {valid_csv}")
            dataset_args.update({
                'rvf10k_train_csv': train_csv,
                'rvf10k_valid_csv': valid_csv,
                'rvf10k_test_csv': test_csv,
                'rvf10k_root_dir': self.dataset_dir  # استفاده از dataset_dir به عنوان rvf10k_root_dir
            })
        elif self.dataset_mode == '140k':
            train_csv = self.args.realfake140k_train_csv
            valid_csv = self.args.realfake140k_valid_csv
            test_csv = self.args.realfake140k_test_csv
            if not os.path.exists(train_csv):
                raise FileNotFoundError(f"140k train CSV not found: {train_csv}")
            if not os.path.exists(valid_csv):
                raise FileNotFoundError(f"140k valid CSV not found: {valid_csv}")
            dataset_args.update({
                'realfake140k_train_csv': train_csv,
                'realfake140k_valid_csv': valid_csv,
                'realfake140k_test_csv': test_csv,
                'realfake140k_root_dir': self.dataset_dir
            })
        elif self.dataset_mode == '200k':
            train_csv = self.args.realfake200k_train_csv
            valid_csv = self.args.realfake200k_val_csv
            test_csv = self.args.realfake200k_test_csv
            if not os.path.exists(train_csv):
                raise FileNotFoundError(f"200k train CSV not found: {train_csv}")
            if not os.path.exists(valid_csv):
                raise FileNotFoundError(f"200k valid CSV not found: {valid_csv}")
            dataset_args.update({
                'realfake200k_train_csv': train_csv,
                'realfake200k_val_csv': valid_csv,
                'realfake200k_test_csv': test_csv,
                'realfake200k_root_dir': self.dataset_dir
            })
        elif self.dataset_mode == '190k':
            root_dir = self.args.realfake190k_root_dir
            if not os.path.exists(root_dir):
                raise FileNotFoundError(f"190k root directory not found: {root_dir}")
            dataset_args.update({'realfake190k_root_dir': root_dir})
        elif self.dataset_mode == '330k':
            root_dir = self.args.realfake330k_root_dir
            if not os.path.exists(root_dir):
                raise FileNotFoundError(f"330k root directory not found: {root_dir}")
            dataset_args.update({'realfake330k_root_dir': root_dir})
        else:
            raise ValueError(f"Unknown dataset_mode: {self.dataset_mode}")

        # Print dataset_args for debugging
        if self.rank == 0:
            self.logger.info(f"dataset_args: {dataset_args}")

        # Initialize dataset
        dataset = Dataset_selector(**dataset_args)
        self.train_loader = dataset.loader_train
        self.val_loader = dataset.loader_val
        self.test_loader = dataset.loader_test
        if self.rank == 0:
            self.logger.info("Dataset loaded successfully!")

    def build_model(self):
        """Build and load the student model."""
        if self.rank == 0:
            self.logger.info("==> Building model...")
            self.logger.info("Loading student model")
        if not os.path.exists(self.finetune_student_ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found: {self.finetune_student_ckpt_path}")
        
        self.student = ResNet_50_sparse_hardfakevsreal()
        ckpt_student = torch.load(self.finetune_student_ckpt_path, map_location="cpu", weights_only=True)
        self.student.load_state_dict(ckpt_student["student"])
        if self.rank == 0:
            self.best_prec1_before_finetune = ckpt_student["best_prec1"]
        self.student = self.student.cuda()
        self.student = DDP(self.student, device_ids=['cuda'], find_unused_parameters=True)

    def define_loss(self):
        """Define the loss function."""
        self.ori_loss = nn.BCEWithLogitsLoss()

    def define_optim(self):
        """Define optimizer and scheduler."""
        weight_params = [
            p for n, p in self.student.module.named_parameters()
            if p.requires_grad and "mask" not in n
        ]
        self.finetune_optim_weight = torch.optim.Adamax(
            weight_params,
            lr=self.finetune_lr,
            weight_decay=self.finetune_weight_decay,
            eps=1e-7,
        )
        self.finetune_scheduler_student_weight = scheduler.CosineAnnealingLRWarmup(
            self.finetune_optim_weight,
            T_max=self.finetune_lr_decay_T_max,
            eta_min=self.finetune_lr_decay_eta_min,
            last_epoch=-1,
            warmup_steps=self.finetune_warmup_steps,
            warmup_start_lr=self.finetune_warmup_start_lr,
        )

    def resume_student_ckpt(self):
        """Resume training from a checkpoint."""
        ckpt_student = torch.load(self.finetune_resume, map_location="cpu", weights_only=True)
        self.best_prec1_after_finetune = ckpt_student["best_prec1_after_finetune"]
        self.start_epoch = ckpt_student["start_epoch"]
        self.student.module.load_state_dict(ckpt_student["student"])
        self.finetune_optim_weight.load_state_dict(
            ckpt_student["finetune_optim_weight"]
        )
        self.finetune_scheduler_student_weight.load_state_dict(
            ckpt_student["finetune_scheduler_student_weight"]
        )
        if riesco == 0:
            self.logger.info(f"=> Resuming from epoch {self.start_epoch}...")

    def save_student_ckpt(self, is_best):
        """Save model checkpoint."""
        if self.rank == 0:
            folder = os.path.join(self.result_dir, "student_model")
            if not os.path.exists(folder):
                os.makedirs(folder)

            ckpt_student = {
                "best_prec1_after_finetune": self.best_prec1_after_finetune,
                "start_epoch": self.start_epoch,
                "student": self.student.module.state_dict(),
                "finetune_optim_weight": self.finetune_optim_weight.state_dict(),
                "finetune_scheduler_student_weight": self.finetune_scheduler_student_weight.state_dict(),
            }

            if is_best:
                torch.save(
                    ckpt_student,
                    os.path.join(folder, f"finetune_{self.arch}_sparse_best.pt"),
                )
            torch.save(
                ckpt_student,
                os.path.join(folder, f"finetune_{self.arch}_sparse_last.pt"),
            )

    def reduce_tensor(self, tensor):
        """Reduce tensor across all processes in DDP."""
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.world_size
        return rt

    def finetune(self):
        """Perform finetuning of the student model."""
        self.ori_loss = self.ori_loss.cuda()
        if self.finetune_resume:
            self.resume_student_ckpt()

        if self.rank == 0:
            meter_oriloss = meter.AverageMeter("OriLoss", ":.4e")
            meter_loss = meter.AverageMeter("Loss", ":.4e")
            meter_top1 = meter.AverageMeter("Acc@1", ":6.2f")

        for epoch in range(self.start_epoch + 1, self.finetune_num_epochs + 1):
            self.train_loader.sampler.set_epoch(epoch)
            self.student.module.train()
            self.student.module.ticket = True
            if self.rank == 0:
                meter_oriloss.reset()
                meter_loss.reset()
                meter_top1.reset()
                finetune_lr = (
                    self.finetune_optim_weight.state_dict()["param_groups"][0]["lr"]
                    if epoch > 1
                    else self.finetune_warmup_start_lr
                )

            with tqdm(total=len(self.train_loader), ncols=100) as _tqdm:
                if self.rank == 0:
                    _tqdm.set_description(f"Epoch: {(epoch}/{self.finetune_num_epochs}")
                for images, targets in self.train_loader:
                    self.finetune_optim_weight.zero_grad()
                    images = images.cuda()
                    targets = targets.cuda().float()
                    logits_student, _ = self.student(images)
                    logits_student = logits_student.squeeze(1)
                    ori_loss = self.ori_loss(logits_student, targets)
                    total_loss = ori_loss
                    total_loss.backward()
                    self.finetune_optim_weight.step()

                    preds = (torch.sigmoid(logits_student) > 0.5).float()
                    correct = (preds == targets).sum().item()
                    prec1 = torch.tensor(100. * correct / images.size(0), device=images.device)
                    n = images.size(0)

                    dist.barrier()
                    reduced_ori_loss = self.reduce_tensor(ori_loss)
                    reduced_total_loss = self.reduce_tensor(total_loss)
                    reduced_prec1 = self.reduce_tensor(prec1)

                    if self.rank == 0:
                        meter_oriloss.update(reduced_ori_loss.item(), n)
                        meter_loss.update(reduced_total_loss.item(), n)
                        meter_top1.update(reduced_prec1.item(), n)

                        _tqdm.set_postfix(
                            loss=f"{meter_loss.avg:.4f}",
                            top1=f"{meter_top1.avg:.4f}",
                        )
                        _tqdm.update(1)
                    time.sleep(0.01)

            self.finetune_scheduler_student_weight.step()

            if self.rank == 0:
                self.writer.add_scalar("finetune_train/loss/ori_loss", meter_oriloss.avg, epoch)
                self.writer.add_scalar("finetune_train/loss/total_loss", meter_loss.avg, epoch)
                self.writer.add_scalar("finetune_train/acc/top1", meter_top1.avg, epoch)
                self.writer.add_scalar("finetune_train/lr/lr", finetune_lr, epoch)

                self.logger.info(
                    f"[Finetune_train] Epoch {epoch}: "
                    f"Learning rate {finetune_lr:.6f} "
                    f"Original loss {meter_oriloss.avg:.4f} "
                    f"Total loss {meter_loss.avg:.4f} "
                    f"Accuracy@1 {meter_top1.avg:.2f}"
                )

            if self.rank == 0:
                self.student.module.eval()
                self.student.module.ticket = True
                meter_top1.reset()
                with torch.no_grad():
                    with tqdm(total=len(self.val_loader), ncols=100) as _tqdm:
                        _tqdm.set_description(f"Epoch: {epoch}/{self.finetune_num_epochs}")
                        for images, targets in self.val_loader:
                            images = images.cuda()
                            targets = targets.cuda().float()
                            logits_student, _ = self.student(images)
                            logits_student = logits_student.squeeze(1)
                            preds = (torch.sigmoid(logits_student) > 0.5).float()
                            correct = (preds == targets).sum().item()
                            prec1 = 100. * correct / images.size(0)
                            n = images.size(0)
                            meter_top1.update(prec1, n)
                            _tqdm.set_postfix(top1=f"{meter_top1.avg:.4f}")
                            _tqdm.update(1)
                            time.sleep(0.01)

                self.writer.add_scalar("finetune_val/acc/top1", meter_top1.avg, epoch)
                self.logger.info(
                    f"[Finetune_val] Epoch {epoch}: Accuracy@1 {meter_top1.avg:.2f}"
                )

                masks = [round(m.mask.mean().item(), 2) for m in self.student.module.mask_modules]
                self.logger.info(f"[Average mask] Epoch {epoch}: {masks}")

                self.start_epoch += 1
                if self.best_prec1_after_finetune < meter_top1.avg:
                    self.best_prec1_after_finetune = meter_top1.avg
                    self.save_student_ckpt(True)
                else:
                    self.save_student_ckpt(False)

                self.logger.info(f" => Best Accuracy@1 before finetune: {self.best_prec1_before_finetune}")
                self.logger.info(f" => Best Accuracy@1 after finetune: {self.best_prec1_after_finetune}")

        if self.rank == 0:
            self.logger.info("Finetuning completed!")
            self.logger.info(f"Best Accuracy@1: {self.best_prec1_after_finetune}")
            try:
                (
                    Flops_baseline,
                    Flops,
                    Flops_reduction,
                    Params_baseline,
                    Params,
                    Params_reduction,
                ) = utils.get_flops_and_params(self.args)
                self.logger.info(
                    f"Baseline parameters: {Params_baseline:.2f}M, Parameters: {Params:.2f}M, Parameter reduction: {Params_reduction:.2f}%"
                )
                self.logger.info(
                    f"Baseline FLOPs: {Flops_baseline:.2f}M, FLOPs: {Flops:.2f}M, FLOPs reduction: {Flops_reduction:.2f}%"
                )
            except AttributeError:
                self.logger.warning("Function get_flops_and_params not found in utils. Skipping FLOPs and parameters calculation.")

    def main(self):
        """Main function to orchestrate finetuning process."""
        try:
            self.dist_init()
            self.result_init()
            self.setup_seed()
            self.dataload()
            self.build_model()
            self.define_loss()
            self.define_optim()
            self.finetune()
        finally:
            if self.args.ddp:
                dist.destroy_process_group()

# کلاس FaceDataset (بدون تغییر)
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

# کلاس Dataset_selector (با اصلاح مسیر تصاویر برای rvf10k)
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
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=8,
        pin_memory=True,
        ddp=False,
    ):
        if dataset_mode not in ['hardfake', 'rvf10k', '140k', '200k', '190k', '12.9k', '330k', '125k']:
            raise ValueError("dataset_mode must be one of 'hardfake', 'rvf10k', '140k', '200k', '190k', '12.9k', '330k', or '125k'")

        self.dataset_mode = dataset_mode

        # Set image size
        image_size = (160, 160) if dataset_mode == '125k' else (256, 256) if dataset_mode in ['rvf10k', '140k', '200k', '190k', '12.9k', '330k'] else (300, 300)

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
        img_column = 'path' if dataset_mode in ['rvf10k', '140k', '12.9k'] else 'filename' if dataset_mode in ['200k', '190k', '330k', '125k'] else 'images_id'

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
                return os.path.join(split, folder, img_name)  # اصلاح‌شده برای مطابقت با ساختار /kaggle/input/rvf10k/train/real/...

            train_data['path'] = train_data.apply(lambda row: create_image_path(row, 'train'), axis=1)
            valid_data['path'] = valid_data.apply(lambda row: create_image_path(row, 'valid'), axis=1)

            for path in train_data['path'].head():
                full_path = os.path.join(root_dir, path)
                print(f"Checking train image {full_path}: {'Exists' if os.path.exists(full_path) else 'Not Found'}")
            for path in valid_data['path'].head():
                full_path = os.path.join(root_dir, path)
                print(f"Checking valid image {full_path}: {'Exists' if os.path.exists(full_path) else 'Not Found'}")

            if rvf10k_test_csv and os.path.exists(rvf10k_test_csv):
                test_data = pd.read_csv(rvf10k_test_csv)
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

                for img_path in glob.glob(os.path.join(real_path, 'real_*.jpg')):
                    data['filename'].append(os.path.relpath(img_path, root_dir))
                    data['label'].append(1)  # Real = 1

                for img_path in glob.glob(os.path.join(fake_path, 'fake_*.jpg')):
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

                for img_path in glob.glob(os.path.join(real_path, '*.jpg')):
                    data['filename'].append(os.path.relpath(img_path, root_dir))
                    data['label'].append(1)  # Real = 1

                for img_path in glob.glob(os.path.join(fake_path, '*.jpg')):
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

    if args.dataset_mode == 'hardfake':
        if not args.hardfake_csv_file or not args.hardfake_root_dir:
            raise ValueError("hardfake_csv_file and hardfake_root_dir must be provided for hardfake dataset")
        dataset = Dataset_selector(
            dataset_mode=args.dataset_mode,
            hardfake_csv_file=args.hardfake_csv_file,
            hardfake_root_dir=args.hardfake_root_dir,
            train_batch_size=args.batch_size,
            eval_batch_size=args.batch_size,
            ddp=True,
        )
    elif args.dataset_mode == 'rvf10k':
        if not args.rvf10k_train_csv or not args.rvf10k_valid_csv or not args.rvf10k_root_dir:
            raise ValueError("rvf10k_train_csv, rvf10k_valid_csv, and rvf10k_root_dir must be provided for rvf10k dataset")
        dataset = Dataset_selector(
            dataset_mode=args.dataset_mode,
            rvf10k_train_csv=args.rvf10k_train_csv,
            rvf10k_valid_csv=args.rvf10k_valid_csv,
            rvf10k_test_csv=args.rvf10k_test_csv,
            rvf10k_root_dir=args.rvf10k_root_dir,
            train_batch_size=args.batch_size,
            eval_batch_size=args.batch_size,
            ddp=True,
        )
    elif args.dataset_mode == '140k':
        if not args.realfake140k_train_csv or not args.realfake140k_valid_csv or not args.realfake140k_test_csv or not args.realfake140k_root_dir:
            raise ValueError("realfake140k_train_csv, realfake140k_valid_csv, realfake140k_test_csv, and realfake140k_root_dir must be provided for 140k dataset")
        dataset = Dataset_selector(
            dataset_mode=args.dataset_mode,
            realfake140k_train_csv=args.realfake140k_train_csv,
            realfake140k_valid_csv=args.realfake140k_valid_csv,
            realfake140k_test_csv=args.realfake140k_test_csv,
            realfake140k_root_dir=args.realfake140k_root_dir,
            train_batch_size=args.batch_size,
            eval_batch_size=args.batch_size,
            ddp=True,
        )
    elif args.dataset_mode == '200k':
        if not args.realfake200k_train_csv or not args.realfake200k_val_csv or not args.realfake200k_test_csv or not args.realfake200k_root_dir:
            raise ValueError("realfake200k_train_csv, realfake200k_val_csv, realfake200k_test_csv, and realfake200k_root_dir must be provided for 200k dataset")
        dataset = Dataset_selector(
            dataset_mode=args.dataset_mode,
            realfake200k_train_csv=args.realfake200k_train_csv,
            realfake200k_val_csv=args.realfake200k_val_csv,
            realfake200k_test_csv=args.realfake200k_test_csv,
            realfake200k_root_dir=args.realfake200k_root_dir,
            train_batch_size=args.batch_size,
            eval_batch_size=args.batch_size,
            ddp=True,
        )
    elif args.dataset_mode == '190k':
        if not args.data_dir:
            raise ValueError("data_dir must be provided for 190k dataset")
        dataset = Dataset_selector(
            dataset_mode=args.dataset_mode,
            realfake190k_root_dir=args.data_dir,
            train_batch_size=args.batch_size,
            eval_batch_size=args.batch_size,
            ddp=True,
        )
    elif args.dataset_mode == '12.9k':
        if not args.dataset_12_9k_csv_file or not args.dataset_12_9k_root_dir:
            raise ValueError("dataset_12_9k_csv_file and dataset_12_9k_root_dir must be provided for 12.9k dataset")
        dataset = Dataset_selector(
            dataset_mode=args.dataset_mode,
            dataset_12_9k_csv_file=args.dataset_12_9k_csv_file,
            dataset_12_9k_root_dir=args.dataset_12_9k_root_dir,
            train_batch_size=args.batch_size,
            eval_batch_size=args.batch_size,
            ddp=True,
        )
    elif args.dataset_mode == '330k':
        if not args.data_dir:
            raise ValueError("data_dir must be provided for 330k dataset")
        dataset = Dataset_selector(
            dataset_mode=args.dataset_mode,
            realfake330k_root_dir=args.data_dir,
            train_batch_size=args.batch_size,
            eval_batch_size=args.batch_size,
            ddp=True,
        )
    elif args.dataset_mode == '125k':
        if not args.data_dir:
            raise ValueError("data_dir must be provided for 125k dataset")
        dataset = Dataset_selector(
            dataset_mode=args.dataset_mode,
            realfake125k_root_dir=args.data_dir,
            train_batch_size=args.batch_size,
            eval_batch_size=args.batch_size,
            ddp=True,
        )
