"""
Main script for satellite image classification.
"""

import os
import torch
import argparse
from torchvision import transforms as T

from dataset import get_dataloaders
from train import create_model, Trainer
from visualization import (
    visualize_dataset, 
    plot_class_distribution, 
    plot_learning_curves
)
from inference import ModelEvaluator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Satellite Image Classification with PyTorch'
    )
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Directory to save outputs')
    parser.add_argument('--model_name', type=str, default='rexnet_150',
                       help='Model architecture name')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=2024,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    return parser.parse_args()


def setup_output_dirs(output_dir):
    """Create output directories."""
    dirs = {
        'models': os.path.join(output_dir, 'models'),
        'visualizations': os.path.join(output_dir, 'visualizations'),
        'results': os.path.join(output_dir, 'results')
    }
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    return dirs


def main():
    """Main training and evaluation pipeline."""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Setup output directories
    dirs = setup_output_dirs(args.output_dir)
    print(f"Output directories created at: {args.output_dir}")
    
    # Define transformations
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transforms = T.Compose([
        T.Resize((args.image_size, args.image_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])
    
    # Load data
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        root=args.data_dir,
        transformations=transforms,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {list(class_names.keys())}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Visualize dataset
    print("\n" + "="*60)
    print("DATASET VISUALIZATION")
    print("="*60)
    
    print("Visualizing training samples...")
    visualize_dataset(
        train_loader.dataset,
        num_images=20,
        rows=4,
        class_names=list(class_names.keys()),
        save_path=os.path.join(dirs['visualizations'], 'train_samples.png')
    )
    
    print("Plotting class distribution...")
    plot_class_distribution(
        root=args.data_dir,
        transformations=transforms,
        save_path=os.path.join(dirs['visualizations'], 'class_distribution.png')
    )
    
    # Create model
    print("\n" + "="*60)
    print("MODEL SETUP")
    print("="*60)
    model = create_model(
        model_name=args.model_name,
        num_classes=num_classes,
        pretrained=True
    )
    print(f"Model: {args.model_name}")
    print(f"Device: {args.device}")
    
    # Train model
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        device=args.device,
        learning_rate=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        save_dir=dirs['models'],
        model_name='satellite'
    )
    
    metrics = trainer.train()
    
    # Plot learning curves
    print("\n" + "="*60)
    print("LEARNING CURVES")
    print("="*60)
    plot_learning_curves(
        train_losses=metrics['train_losses'],
        val_losses=metrics['val_losses'],
        train_accs=metrics['train_accs'],
        val_accs=metrics['val_accs'],
        train_f1s=metrics['train_f1s'],
        val_f1s=metrics['val_f1s'],
        save_dir=dirs['visualizations']
    )
    
    # Evaluation
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
    # Load best model
    model.load_state_dict(
        torch.load(os.path.join(dirs['models'], 'satellite_best_model.pth'))
    )
    
    # Get layers for GradCAM (adjust based on model architecture)
    if hasattr(model, 'features'):
        final_conv = model.features[-1]
    else:
        # For other architectures, you may need to adjust this
        final_conv = list(model.children())[-2]
    
    if hasattr(model, 'head') and hasattr(model.head, 'fc'):
        fc_params = list(model.head.fc.parameters())
    else:
        fc_params = list(model.fc.parameters()) if hasattr(model, 'fc') else []
    
    # Run evaluation
    evaluator = ModelEvaluator(
        model=model,
        test_loader=test_loader,
        device=args.device,
        class_names=class_names,
        final_conv_layer=final_conv,
        fc_params=fc_params,
        image_size=args.image_size
    )
    
    results = evaluator.run_full_evaluation(save_dir=dirs['results'])
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print(f"Final Test Accuracy: {results['accuracy']:.4f}")
    print(f"All outputs saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
