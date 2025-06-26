import torch
import argparse
from model.student.ResNet_sparse import ResNet_50_sparse_hardfakevsreal
from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
from thop import profile

# Base FLOPs and parameters for each dataset
Flops_baselines = {
    "ResNet_50": {
        "hardfakevsreal": 7700.0,
        "rvf10k": 5000.0,
        "140k": 5390.0,
        "200k": 5390.0,
        "190k": 5390.0,
        "330k": 5390.0,  
    }
}
Params_baselines = {
    "ResNet_50": {
        "hardfakevsreal": 14.97,
        "rvf10k": 23.51,
        "140k": 23.51,
        "200k": 23.51,
        "190k": 23.51,
        "330k": 23.51,  
    }
}
image_sizes = {
    "hardfakevsreal": 300,
    "rvf10k": 256,
    "140k": 256,
    "200k": 256,
    "190k": 256,
    "330k": 256,  
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_mode",
        type=str,
        default="hardfake",
        choices=("hardfake", "rvf10k", "140k", "200k", "190k", "330k"),
        help="The type of dataset",
    )
    parser.add_argument(
        "--sparsed_student_ckpt_path",
        type=str,
        default=None,
        help="The path where to load the sparsed student ckpt",
    )
    return parser.parse_args()


def get_flops_and_params(args):
    # Map dataset_mode to dataset_type
    dataset_type = {
        "hardfake": "hardfakevsreal",
        "rvf10k": "rvf10k",
        "140k": "140k",
        "200k": "200k",
        "190k": "190k",
        "330k": "330k" 
    }[args.dataset_mode]

    # Load sparse student model to extract masks
    if args.dataset_mode == "hardfake":
        student = ResNet_50_sparse_hardfakevsreal()
    elif args.dataset_mode == "rvf10k":
        student = ResNet_50_sparse_hardfakevsreal()
    elif args.dataset_mode == "140k":
        student = ResNet_50_sparse_hardfakevsreal()
    elif args.dataset_mode == "200k":
        student =ResNet_50_sparse_hardfakevsreal()
    elif args.dataset_mode == "190k":
        student = ResNet_50_sparse_hardfakevsreal()
    elif args.dataset_mode == "330k":
        student = ResNet_50_sparse_hardfakevsreal()

    ckpt_student = torch.load(args.sparsed_student_ckpt_path, map_location="cpu", weights_only=True)
    student.load_state_dict(ckpt_student["student"])

    # Extract masks
    mask_weights = [m.mask_weight for m in student.mask_modules]
    masks = [
        torch.argmax(mask_weight, dim=1).squeeze(1).squeeze(1)
        for mask_weight in mask_weights
    ]

    # Load pruned model with masks
    if args.dataset_mode == "hardfake":
        pruned_model = ResNet_50_pruned_hardfakevsreal(masks=masks)
    elif args.dataset_mode == "rvf10k":
        pruned_model =  ResNet_50_sparse_hardfakevsreal(masks=masks)
    elif args.dataset_mode == "140k":
        pruned_model = ResNet_50_sparse_hardfakevsreal(masks=masks)
    elif args.dataset_mode == "200k":
        pruned_model =  ResNet_50_sparse_hardfakevsreal(masks=masks)
    elif args.dataset_mode == "190k":
        pruned_model =  ResNet_50_sparse_hardfakevsreal(masks=masks)
    elif args.dataset_mode == "330k":
        pruned_model =  ResNet_50_sparse_hardfakevsreal(masks=masks)
    
    # Set input size based on dataset
    input = torch.rand([1, 3, image_sizes[dataset_type], image_sizes[dataset_type]])
    Flops, Params = profile(pruned_model, inputs=(input,), verbose=False)

    # Use dataset-specific baseline values
    Flops_baseline = Flops_baselines["ResNet_50"][dataset_type]
    Params_baseline = Params_baselines["ResNet_50"][dataset_type]

    Flops_reduction = (
        (Flops_baseline - Flops / (10**6)) / Flops_baseline * 100.0
    )
    Params_reduction = (
        (Params_baseline - Params / (10**6)) / Params_baseline * 100.0
    )
    return (
        Flops_baseline,
        Flops / (10**6),
        Flops_reduction,
        Params_baseline,
        Params / (10**6),
        Params_reduction,
    )

def main():
    args = parse_args()

    # Run for all datasets
    for dataset_mode in ["hardfake", "rvf10k", "140k", "200k", "190k", "330k"]:
        print(f"\nEvaluating for dataset: {dataset_mode}")
        args.dataset_mode = dataset_mode
        (
            Flops_baseline,
            Flops,
            Flops_reduction,
            Params_baseline,
            Params,
            Params_reduction,
        ) = get_flops_and_params(args=args)
        print(
            "Params_baseline: %.2fM, Params: %.2fM, Params reduction: %.2f%%"
            % (Params_baseline, Params, Params_reduction)
        )
        print(
            "Flops_baseline: %.2fM, Flops: %.2fM, Flops reduction: %.2f%%"
            % (Flops_baseline, Flops, Flops_reduction)
        )

if __name__ == "__main__":
    main()
