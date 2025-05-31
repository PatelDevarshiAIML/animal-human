import os
import shutil
import random
from pathlib import Path

def split_dataset(images_dir, labels_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Split images and labels into train/val/test sets for YOLOv8
    Each split contains both images/ and labels/ folders
    
    Args:
        images_dir: Path to directory containing all images
        labels_dir: Path to directory containing all label files
        output_dir: Path to output directory for split dataset
        train_ratio: Proportion for training set (default: 0.7)
        val_ratio: Proportion for validation set (default: 0.2)
        test_ratio: Proportion for test set (default: 0.1)
    """
    
    # Check if ratios sum to 1
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
        raise ValueError("Train, val, and test ratios must sum to 1.0")
    
    # Create output directory structure
    # Each split has its own images/ and labels/ folders
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(images_dir).glob(f'*{ext}'))
        image_files.extend(Path(images_dir).glob(f'*{ext.upper()}'))
    
    # Filter images that have corresponding label files
    valid_pairs = []
    missing_labels = []
    
    for image_path in image_files:
        label_name = image_path.stem + '.txt'
        label_path = Path(labels_dir) / label_name
        
        if label_path.exists():
            valid_pairs.append((image_path, label_path))
        else:
            missing_labels.append(image_path.name)
    
    if missing_labels:
        print(f"Warning: {len(missing_labels)} images have no corresponding labels:")
        for missing in missing_labels[:10]:  # Show first 10
            print(f"  - {missing}")
        if len(missing_labels) > 10:
            print(f"  ... and {len(missing_labels) - 10} more")
    
    print(f"Found {len(valid_pairs)} valid image-label pairs")
    
    if len(valid_pairs) == 0:
        raise ValueError("No valid image-label pairs found!")
    
    # Shuffle the pairs randomly
    random.shuffle(valid_pairs)
    
    # Calculate split indices
    total_files = len(valid_pairs)
    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)
    
    # Split the data
    train_pairs = valid_pairs[:train_end]
    val_pairs = valid_pairs[train_end:val_end]
    test_pairs = valid_pairs[val_end:]
    
    # Copy files to respective directories
    def copy_pairs(pairs, split_name):
        print(f"\nCopying {len(pairs)} files to {split_name} set...")
        for i, (image_path, label_path) in enumerate(pairs):
            # Copy image to split/images/
            dest_image = os.path.join(output_dir, split_name, 'images', image_path.name)
            shutil.copy2(image_path, dest_image)
            
            # Copy label to split/labels/
            dest_label = os.path.join(output_dir, split_name, 'labels', label_path.name)
            shutil.copy2(label_path, dest_label)
            
            if (i + 1) % 100 == 0 or (i + 1) == len(pairs):
                print(f"  Copied {i + 1}/{len(pairs)} files")
    
    # Copy files for each split
    copy_pairs(train_pairs, 'train')
    copy_pairs(val_pairs, 'val')
    copy_pairs(test_pairs, 'test')
    
    # Print summary
    print(f"\n{'='*50}")
    print("DATASET SPLIT SUMMARY")
    print(f"{'='*50}")
    print(f"Total files: {total_files}")
    print(f"Train: {len(train_pairs)} files ({len(train_pairs)/total_files*100:.1f}%)")
    print(f"Val: {len(val_pairs)} files ({len(val_pairs)/total_files*100:.1f}%)")
    print(f"Test: {len(test_pairs)} files ({len(test_pairs)/total_files*100:.1f}%)")
    print(f"\nOutput directory: {output_dir}")

def create_yaml_config(output_dir, class_names):
    """Create dataset.yaml config file for YOLOv8"""
    yaml_file = os.path.join(output_dir, "dataset.yaml")
    
    yaml_content = f"""# YOLOv8 Dataset Configuration
path: {os.path.abspath(output_dir)}
train: train/images
val: val/images
test: test/images

# Classes
nc: {len(class_names)}
names: {class_names}
"""
    
    with open(yaml_file, 'w') as f:
        f.write(yaml_content)
    
    print(f"\nCreated dataset.yaml: {yaml_file}")

def main():
    """Main function to run the dataset splitting"""
    
    # Configuration - Modify these paths according to your setup
    images_dir = "main/new_dataset\images"          # Directory containing all your images
    labels_dir = "main/new_dataset\labels"          # Directory containing all your label files
    output_dir = "main/combined/dataset_2"         # Output directory for split dataset
    
    # Split ratios (must sum to 1.0)
    train_ratio = 0.7  # 70% for training
    val_ratio = 0.2    # 20% for validation
    test_ratio = 0.1   # 10% for testing
    
    # Class names (modify according to your dataset)
    class_names = ["animals"]  # Add your class names here
    
    print("Starting dataset splitting...")
    print(f"Images directory: {images_dir}")
    print(f"Labels directory: {labels_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Split ratios - Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}")
    
    # Check if input directories exist
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not os.path.exists(labels_dir):
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    
    # Set random seed for reproducible splits
    random.seed(42)
    
    # Split the dataset
    split_dataset(images_dir, labels_dir, output_dir, train_ratio, val_ratio, test_ratio)
    
    # Create YAML config file
    create_yaml_config(output_dir, class_names)
    
    print(f"\n{'='*50}")
    print("DATASET SPLITTING COMPLETED!")
    print(f"{'='*50}")
    print("Directory structure created:")
    print("dataset/")
    print("├── train/")
    print("│   ├── images/")
    print("│   └── labels/")
    print("├── val/")
    print("│   ├── images/")
    print("│   └── labels/")
    print("├── test/")
    print("│   ├── images/")
    print("│   └── labels/")
    print("└── dataset.yaml")
    print("\nYour dataset is ready for YOLOv8 training!")
    print("To train: yolo train data=dataset/dataset.yaml model=yolov8n.pt")

if __name__ == "__main__":
    main()