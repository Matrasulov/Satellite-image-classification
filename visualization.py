"""
Visualization utilities for satellite image classification.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms as T
from dataset import SatelliteDataset


def tensor_to_image(tensor, image_type="rgb"):
    """
    Convert normalized tensor to displayable image.
    
    Args:
        tensor: Image tensor
        image_type (str): 'rgb' or 'gray'
    
    Returns:
        numpy array: Image in uint8 format
    """
    gray_inverse = T.Compose([
        T.Normalize(mean=[0.], std=[1/0.5]),
        T.Normalize(mean=[-0.5], std=[1])
    ])
    
    rgb_inverse = T.Compose([
        T.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
        T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
    ])
    
    inv_transform = gray_inverse if image_type == "gray" else rgb_inverse
    
    if image_type == "gray":
        return (inv_transform(tensor) * 255).detach().squeeze().cpu().permute(1, 2, 0).numpy().astype(np.uint8)
    else:
        return (inv_transform(tensor) * 255).detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8)


def visualize_dataset(data, num_images=20, rows=4, image_type="rgb", 
                     class_names=None, save_path=None):
    """
    Visualize random samples from dataset.
    
    Args:
        data: Dataset or DataLoader
        num_images (int): Number of images to display
        rows (int): Number of rows in grid
        image_type (str): 'rgb' or 'gray'
        class_names (dict): Dictionary mapping class indices to names
        save_path (str): Path to save the figure
    """
    cmap = "viridis" if image_type == "gray" else None
    
    plt.figure(figsize=(20, 10))
    indices = [random.randint(0, len(data) - 1) for _ in range(num_images)]
    
    for idx, index in enumerate(indices):
        im, gt = data[index]
        
        plt.subplot(rows, num_images // rows, idx + 1)
        if cmap:
            plt.imshow(tensor_to_image(im, image_type), cmap=cmap)
        else:
            plt.imshow(tensor_to_image(im, image_type))
        plt.axis('off')
        
        if class_names is not None:
            plt.title(f"Class: {class_names[int(gt)]}")
        else:
            plt.title(f"Class: {gt}")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    plt.show()


def plot_class_distribution(root, transformations, save_path=None):
    """
    Plot class distribution bar chart.
    
    Args:
        root (str): Dataset root directory
        transformations: Transforms to apply
        save_path (str): Path to save the figure
    """
    dataset = SatelliteDataset(root=root, transformations=transformations)
    cls_counts = dataset.cls_counts
    
    class_names = list(cls_counts.keys())
    counts = list(cls_counts.values())
    
    fig, ax = plt.subplots(figsize=(20, 10))
    indices = np.arange(len(counts))
    
    bars = ax.bar(indices, counts, width=0.7, color="firebrick")
    
    ax.set_xlabel("Class Names", fontsize=14, color="red")
    ax.set_ylabel("Number of Images", fontsize=14, color="red")
    ax.set_title("Dataset Class Distribution Analysis", fontsize=16, fontweight='bold')
    ax.set_xticks(indices)
    ax.set_xticklabels(class_names, rotation=60, ha='right')
    
    # Add value labels on bars
    for i, (bar, v) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                str(v), ha='center', va='bottom', color='royalblue', 
                fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved class distribution to {save_path}")
    plt.show()


def plot_learning_curves(train_losses, val_losses, train_accs, val_accs, 
                         train_f1s, val_f1s, save_dir=None):
    """
    Plot training and validation metrics.
    
    Args:
        train_losses, val_losses: Loss values per epoch
        train_accs, val_accs: Accuracy values per epoch
        train_f1s, val_f1s: F1 score values per epoch
        save_dir (str): Directory to save figures
    """
    epochs_range = range(1, len(train_losses) + 1)
    
    # Loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, train_losses, label='Train Loss', c='red', marker='o')
    plt.plot(epochs_range, val_losses, label='Validation Loss', c='blue', marker='s')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss Values', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.xticks(epochs_range)
    plt.legend()
    plt.grid(alpha=0.3)
    if save_dir:
        plt.savefig(f"{save_dir}/loss_curve.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Accuracy plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, train_accs, label='Train Accuracy', c='orangered', marker='o')
    plt.plot(epochs_range, val_accs, label='Validation Accuracy', c='darkgreen', marker='s')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy Scores', fontsize=12)
    plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    plt.xticks(epochs_range)
    plt.legend()
    plt.grid(alpha=0.3)
    if save_dir:
        plt.savefig(f"{save_dir}/accuracy_curve.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # F1 Score plot
    plt.figure(figsize=(10, 5))
    train_f1s_cpu = [f1.cpu() if hasattr(f1, 'cpu') else f1 for f1 in train_f1s]
    val_f1s_cpu = [f1.cpu() if hasattr(f1, 'cpu') else f1 for f1 in val_f1s]
    plt.plot(epochs_range, train_f1s_cpu, label='Train F1 Score', c='blueviolet', marker='o')
    plt.plot(epochs_range, val_f1s_cpu, label='Validation F1 Score', c='crimson', marker='s')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('F1 Scores', fontsize=12)
    plt.title('Training and Validation F1 Score', fontsize=14, fontweight='bold')
    plt.xticks(epochs_range)
    plt.legend()
    plt.grid(alpha=0.3)
    if save_dir:
        plt.savefig(f"{save_dir}/f1_score_curve.png", dpi=300, bbox_inches='tight')
    plt.show()
