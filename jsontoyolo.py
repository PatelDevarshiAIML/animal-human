import json
import os
from pathlib import Path

def convert_ndjson_to_yolov8(ndjson_file, output_dir, class_mapping=None):
    """
    Convert NDJSON annotations to YOLOv8 format
    Creates separate .txt label file for each image
    
    Args:
        ndjson_file: Path to the NDJSON file
        output_dir: Directory to save YOLOv8 label files
        class_mapping: Dictionary mapping class names to class IDs
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Default class mapping if not provided
    if class_mapping is None:
        class_mapping = {"animals": 0}
    
    processed_images = 0
    total_annotations = 0
    
    # Process each line in NDJSON file
    with open(ndjson_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
                
            try:
                data = json.loads(line.strip())
                
                # Extract image info
                external_id = data['data_row']['external_id']
                image_name = Path(external_id).stem  # Remove extension (.jpg, .png, etc.)
                
                # Get image dimensions
                width = data['media_attributes']['width']
                height = data['media_attributes']['height']
                
                # Create label file path (same name as image but .txt extension)
                label_file = os.path.join(output_dir, f"{image_name}.txt")
                
                # Extract annotations
                yolo_annotations = []
                
                # Check if there are any projects with labels
                projects = data.get('projects', {})
                for project_id, project_data in projects.items():
                    labels = project_data.get('labels', [])
                    
                    for label in labels:
                        annotations = label.get('annotations', {})
                        objects = annotations.get('objects', [])
                        
                        for obj in objects:
                            if obj.get('annotation_kind') == 'ImageBoundingBox':
                                # Get bounding box coordinates
                                bbox = obj['bounding_box']
                                x_left = bbox['left']
                                y_top = bbox['top']
                                box_width = bbox['width']
                                box_height = bbox['height']
                                
                                # Convert to YOLO format (normalized center coordinates)
                                x_center = (x_left + box_width / 2) / width
                                y_center = (y_top + box_height / 2) / height
                                norm_width = box_width / width
                                norm_height = box_height / height
                                
                                # Get class ID
                                class_name = obj.get('value', obj.get('name', 'unknown'))
                                class_id = class_mapping.get(class_name, 0)
                                
                                # Create YOLO annotation line
                                yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
                                yolo_annotations.append(yolo_line)
                                total_annotations += 1
                
                # Write annotations to individual label file
                with open(label_file, 'w') as label_f:
                    if yolo_annotations:
                        label_f.write('\n'.join(yolo_annotations))
                    else:
                        # Create empty file for images with no annotations
                        pass
                
                processed_images += 1
                print(f"Created: {label_file} ({len(yolo_annotations)} annotations)")
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON line: {e}")
            except Exception as e:
                print(f"Error processing line: {e}")
    
    print(f"\nConversion complete!")
    print(f"Processed {processed_images} images")
    print(f"Total annotations: {total_annotations}")

def create_classes_file(output_dir, class_mapping):
    """Create classes.txt file with class names for YOLOv8"""
    classes_file = os.path.join(output_dir, "classes.txt")
    
    # Sort classes by ID
    sorted_classes = sorted(class_mapping.items(), key=lambda x: x[1])
    
    with open(classes_file, 'w') as f:
        for class_name, class_id in sorted_classes:
            f.write(f"{class_name}\n")
    
    print(f"Created classes file: {classes_file}")

def create_yaml_config(output_dir, class_mapping, train_path="", val_path=""):
    """Create dataset.yaml config file for YOLOv8 training"""
    yaml_file = os.path.join(output_dir, "dataset.yaml")
    
    # Sort classes by ID
    sorted_classes = sorted(class_mapping.items(), key=lambda x: x[1])
    class_names = [name for name, _ in sorted_classes]
    
    yaml_content = f"""# Dataset configuration for YOLOv8
path: .  # dataset root dir
train: {train_path or 'images/train'}  # train images (relative to 'path')
val: {val_path or 'images/val'}    # val images (relative to 'path')

# Classes
nc: {len(class_names)}  # number of classes
names: {class_names}  # class names
"""
    
    with open(yaml_file, 'w') as f:
        f.write(yaml_content)
    
    print(f"Created YAML config: {yaml_file}")

# Example usage
if __name__ == "__main__":
    # Define your class mapping (add more classes as needed)
    class_mapping = {
        "animals": 0,
        "humans": 1,
        # "vehicle": 2,
    }
    
    # File paths
    ndjson_file = "main\maharshi_industries\combined\sample_dataset.ndjson"  # Your NDJSON file
    output_dir = "yolo_labels_2"  # Output directory for label files
    
    print("Converting NDJSON to YOLOv8 format...")
    print(f"Input file: {ndjson_file}")
    print(f"Output directory: {output_dir}")
    print(f"Class mapping: {class_mapping}")
    print("-" * 50)
    
    # Convert annotations
    convert_ndjson_to_yolov8(ndjson_file, output_dir, class_mapping)
    
    # Create additional files
    create_classes_file(output_dir, class_mapping)
    create_yaml_config(output_dir, class_mapping)
    
    print("\nFiles created:")
    print("- Individual .txt label files for each image")
    print("- classes.txt (class names)")
    print("- dataset.yaml (YOLOv8 config)")
    print("\nNext steps:")
    print("1. Organize your images into train/val folders")
    print("2. Place corresponding .txt files in labels/train and labels/val")
    print("3. Update paths in dataset.yaml")
    print("4. Train with: yolo train data=dataset.yaml model=yolov8n.pt")