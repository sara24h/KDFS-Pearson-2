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
        pruned_model = ResNet_50_pruned_rvf10k(masks=masks)
    elif args.dataset_mode == "140k":
        pruned_model = ResNet_50_pruned_140k(masks=masks)
    elif args.dataset_mode == "200k":
        pruned_model = ResNet_50_pruned_200k(masks=masks)
    elif args.dataset_mode == "190k":
        pruned_model = ResNet_50_pruned_190k(masks=masks)
    elif args.dataset_mode == "330k":
        pruned_model = ResNet_50_pruned_330k(masks=masks)
    
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
