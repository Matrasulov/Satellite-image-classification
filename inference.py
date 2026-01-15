"""
Inference and evaluation utilities with GradCAM visualization.
"""

import random
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from visualization import tensor_to_image


class SaveFeatures:
    """Hook to extract feature maps from a layer."""
    
    features = None
    
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    
    def hook_fn(self, module, input, output):
        self.features = output.cpu().data.numpy()
    
    def remove(self):
        self.hook.remove()


def get_gradcam(conv_features, linear_weights, class_idx):
    """
    Generate GradCAM heatmap.
    
    Args:
        conv_features: Convolutional feature maps
        linear_weights: Fully connected layer weights
        class_idx (int): Target class index
    
    Returns:
        numpy array: Normalized heatmap
    """
    batch_size, channels, height, width = conv_features.shape
    cam = linear_weights[class_idx].dot(
        conv_features[0, :, :, :].reshape((channels, height * width))
    )
    cam = cam.reshape(height, width)
    
    # Normalize
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
    return cam


class ModelEvaluator:
    """Evaluate model and generate visualizations."""
    
    def __init__(self, model, test_loader, device, class_names, 
                 final_conv_layer, fc_params, image_size=224):
        """
        Args:
            model: Trained PyTorch model
            test_loader: Test dataloader
            device (str): Device to use
            class_names (dict): Class index to name mapping
            final_conv_layer: Final convolutional layer for GradCAM
            fc_params: Fully connected layer parameters
            image_size (int): Image size for heatmap resizing
        """
        self.model = model.to(device).eval()
        self.test_loader = test_loader
        self.device = device
        self.class_names = class_names
        self.image_size = image_size
        
        # GradCAM setup
        self.activated_features = SaveFeatures(final_conv_layer)
        self.fc_weights = np.squeeze(fc_params[0].cpu().data.numpy())
    
    def evaluate(self):
        """
        Evaluate model on test set.
        
        Returns:
            tuple: (predictions, ground_truths, images, accuracy)
        """
        predictions = []
        ground_truths = []
        images = []
        correct = 0
        
        print("Evaluating model on test set...")
        with torch.no_grad():
            for batch in tqdm(self.test_loader):
                image, label = batch[0].to(self.device), batch[1].to(self.device)
                
                output = self.model(image)
                pred_class = torch.argmax(output, dim=1)
                
                correct += (pred_class == label).sum().item()
                
                images.append(image)
                predictions.append(pred_class.item())
                ground_truths.append(label.item())
        
        accuracy = correct / len(self.test_loader.dataset)
        print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        return predictions, ground_truths, images, accuracy
    
    def visualize_gradcam(self, predictions, ground_truths, images, 
                         num_images=20, rows=4, save_path=None):
        """
        Visualize predictions with GradCAM heatmaps.
        
        Args:
            predictions: List of predicted classes
            ground_truths: List of true labels
            images: List of images
            num_images (int): Number of images to visualize
            rows (int): Number of rows in grid
            save_path (str): Path to save figure
        """
        plt.figure(figsize=(20, 10))
        indices = random.sample(range(len(images)), min(num_images, len(images)))
        
        for idx, image_idx in enumerate(indices):
            image = images[image_idx].squeeze()
            pred_idx = predictions[image_idx]
            true_idx = ground_truths[image_idx]
            
            # Generate GradCAM heatmap
            heatmap = get_gradcam(
                self.activated_features.features, 
                self.fc_weights, 
                pred_idx
            )
            
            # Plot
            plt.subplot(rows, num_images // rows, idx + 1)
            plt.imshow(tensor_to_image(image))
            plt.imshow(
                cv2.resize(heatmap, (self.image_size, self.image_size), 
                          interpolation=cv2.INTER_LINEAR),
                alpha=0.4, 
                cmap='jet'
            )
            plt.axis('off')
            
            # Title with color coding
            true_name = list(self.class_names.keys())[true_idx]
            pred_name = list(self.class_names.keys())[pred_idx]
            color = 'green' if true_idx == pred_idx else 'red'
            plt.title(f"GT: {true_name}\nPred: {pred_name}", 
                     color=color, fontsize=10, fontweight='bold')
        
        plt.suptitle('GradCAM Visualizations', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved GradCAM visualizations to {save_path}")
        plt.show()
    
    def plot_confusion_matrix(self, predictions, ground_truths, save_path=None):
        """
        Plot confusion matrix.
        
        Args:
            predictions: List of predicted classes
            ground_truths: List of true labels
            save_path (str): Path to save figure
        """
        cm = confusion_matrix(ground_truths, predictions)
        class_labels = list(self.class_names.keys())
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_labels, 
            yticklabels=class_labels,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Class', fontsize=12, fontweight='bold')
        plt.ylabel('True Class', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved confusion matrix to {save_path}")
        plt.show()
    
    def run_full_evaluation(self, save_dir=None):
        """
        Run complete evaluation pipeline.
        
        Args:
            save_dir (str): Directory to save outputs
        """
        # Evaluate
        predictions, ground_truths, images, accuracy = self.evaluate()
        
        # Generate visualizations
        gradcam_path = f"{save_dir}/gradcam_results.png" if save_dir else None
        cm_path = f"{save_dir}/confusion_matrix.png" if save_dir else None
        
        self.visualize_gradcam(
            predictions, ground_truths, images, 
            num_images=20, rows=4, save_path=gradcam_path
        )
        
        self.plot_confusion_matrix(
            predictions, ground_truths, save_path=cm_path
        )
        
        return {
            'accuracy': accuracy,
            'predictions': predictions,
            'ground_truths': ground_truths
        }
