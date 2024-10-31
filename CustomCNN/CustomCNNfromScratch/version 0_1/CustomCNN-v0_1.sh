#!/bin/bash

# Set default values
CSV_PATH="/home/zhl/regression_and_prediction/features/10w.csv"
IMAGE_DIR="/home/zhl/Python-AI/0930_age_prediction/log_image"
BATCH_SIZE=128
NUM_EPOCHS=400
LEARNING_RATE=0.001
WEIGHT_DECAY=0.0001
FOCAL_GAMMA=1.0
T0=10
T_MULT=2
MIN_LR=1e-4
NUM_WORKERS=4
SEED=42
SAVE_DIR="/home/zhl/regression_and_prediction/2024-10-30"

# Display usage information
show_usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --csv_path PATH        Path to CSV file (default: $CSV_PATH)"
    echo "  --image_dir PATH       Path to image directory (default: $IMAGE_DIR)"
    echo "  --batch_size SIZE      Batch size (default: $BATCH_SIZE)"
    echo "  --num_epochs NUM       Number of epochs (default: $NUM_EPOCHS)"
    echo "  --learning_rate RATE   Learning rate (default: $LEARNING_RATE)"
    echo "  --weight_decay DECAY   Weight decay (default: $WEIGHT_DECAY)"
    echo "  --focal_gamma GAMMA    Focal loss gamma (default: $FOCAL_GAMMA)"
    echo "  --t0 T0               T0 parameter (default: $T0)"
    echo "  --t_mult T_MULT       T_mult parameter (default: $T_MULT)"
    echo "  --min_lr MIN_LR       Minimum learning rate (default: $MIN_LR)"
    echo "  --num_workers NUM      Number of workers (default: $NUM_WORKERS)"
    echo "  --seed NUM             Random seed (default: $SEED)"
    echo "  -h, --help            Show this help message"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --csv_path)
            CSV_PATH="$2"
            shift 2
            ;;
        --image_dir)
            IMAGE_DIR="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --weight_decay)
            WEIGHT_DECAY="$2"
            shift 2
            ;;
        --focal_gamma)
            FOCAL_GAMMA="$2"
            shift 2
            ;;
        --t0)
            T0="$2"
            shift 2
            ;;
        --t_mult)
            T_MULT="$2"
            shift 2
            ;;
        --min_lr)
            MIN_LR="$2"
            shift 2
            ;;
        --num_workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Error: Unknown parameter $1"
            show_usage
            exit 1
            ;;
    esac
done

# Check prerequisites
check_prerequisites() {
    # Check CSV file
    if [ ! -f "$CSV_PATH" ]; then
        echo "Error: CSV file not found: $CSV_PATH"
        exit 1
    fi

    # Check image directory
    if [ ! -d "$IMAGE_DIR" ]; then
        echo "Error: Image directory not found: $IMAGE_DIR"
        exit 1
    fi

    # Check and create save directory
    if [ ! -d "$SAVE_DIR" ]; then
        echo "Creating save directory: $SAVE_DIR"
        mkdir -p "$SAVE_DIR"
    fi
}

# Print training configuration
print_config() {
    echo "=== Training Configuration ==="
    echo "CSV Path:        $CSV_PATH"
    echo "Image Directory: $IMAGE_DIR"
    echo "Batch Size:      $BATCH_SIZE"
    echo "Num Epochs:      $NUM_EPOCHS"
    echo "Learning Rate:   $LEARNING_RATE"
    echo "Weight Decay:    $WEIGHT_DECAY"
    echo "Focal Gamma:     $FOCAL_GAMMA"
    echo "T0:             $T0"
    echo "T_mult:         $T_MULT"
    echo "Min LR:         $MIN_LR"
    echo "Num Workers:     $NUM_WORKERS"
    echo "Random Seed:     $SEED"
    echo "==========================="
}

# Run training
run_training() {
    echo "Starting training..."

    # Record start time
    start_time=$(date +%s)

    # Run Python training script with correct arguments
    python /home/zhl/regression_and_prediction/2024-10-30/CustomCNN_1030.py \
        --csv_path "$CSV_PATH" \
        --image_dir "$IMAGE_DIR" \
        --batch_size "$BATCH_SIZE" \
        --num_epochs "$NUM_EPOCHS" \
        --learning_rate "$LEARNING_RATE" \
        --weight_decay "$WEIGHT_DECAY" \
        --focal_gamma "$FOCAL_GAMMA" \
        --t0 "$T0" \
        --t_mult "$T_MULT" \
        --min_lr "$MIN_LR" \
        --num_workers "$NUM_WORKERS" \
        --seed "$SEED"

    # Check if training was successful
    if [ $? -eq 0 ]; then
        # Calculate training duration
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        hours=$((duration / 3600))
        minutes=$(( (duration % 3600) / 60 ))
        seconds=$((duration % 60))

        echo "Training completed successfully!"
        echo "Total training time: ${hours}h ${minutes}m ${seconds}s"
    else
        echo "Error occurred during training!"
        exit 1
    fi
}

# # Main function
# main() {
#     # Check environment and dependencies
#     check_prerequisites

#     # Print configuration
#     print_config

#     # Ask for user confirmation
#     read -p "Start training? [y/N] " response
#     case "$response" in
#         [yY][eE][sS]|[yY])
#             run_training
#             ;;
#         *)
#             echo "Training cancelled"
#             exit 0
#             ;;
#     esac
# }

# # Run main function
# main

# Main function
main() {
    # 添加自动确认参数
    AUTO_CONFIRM=${AUTO_CONFIRM:-false}

    # Check environment and dependencies
    check_prerequisites

    # Print configuration
    print_config

    if [ "$AUTO_CONFIRM" = true ]; then
        # 自动确认
        run_training
    else
        # 手动确认
        read -p "Start training? [y/N] " response
        case "$response" in
            [yY][eE][sS]|[yY])
                run_training
                ;;
            *)
                echo "Training cancelled"
                exit 0
                ;;
        esac
    fi
}

# Run main function
main