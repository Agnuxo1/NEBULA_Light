"""
NEBULA v4.0 Dataset Loader for Grand X-Ray Slam Division A
========================================================
Processing 107,374 radiographic images with CUDA raytracing
Following NEBULA CREDO: Step by step, with calm precision

Author: NEBULA Team - Francisco Angulo de Lafuente & Claude AGI
Dataset: Grand X-Ray Slam Division A - Thoracic Condition Detection
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from pathlib import Path
import logging
from tqdm import tqdm
import sys

# Import NEBULA system
sys.path.append('C:\\SANTO_GRIAL')
from NEBULA_LUZ import NEBULATrainerV4, NEBULAControlPanelV4, DEVICE

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GrandXRayDataset(Dataset):
    """
    Dataset loader for Grand X-Ray Slam Division A
    Handles 107,374 thoracic radiographs with 14 medical conditions
    """
    
    def __init__(self, csv_file: str, image_dir: str, transform=None, max_samples=None):
        """
        Initialize dataset
        
        Args:
            csv_file: Path to train1.csv
            image_dir: Path to train1/ directory with images
            transform: Image transformations
            max_samples: Limit samples for testing (None = all samples)
        """
        self.csv_file = csv_file
        self.image_dir = Path(image_dir)
        self.transform = transform
        
        logger.info(f"Loading dataset from {csv_file}")
        self.data = pd.read_csv(csv_file)
        
        if max_samples:
            self.data = self.data.head(max_samples)
            logger.info(f"Limited to {max_samples} samples for testing")
        
        logger.info(f"Dataset loaded: {len(self.data)} samples")
        
        # 14 medical conditions (matching NEBULA v4.0)
        self.conditions = [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
            'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
            'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
            'Pneumothorax', 'Support Devices'
        ]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load image
        image_path = self.image_dir / row['Image_name']
        
        try:
            image = Image.open(image_path).convert('L')  # Convert to grayscale
            
            # Resize to NEBULA v4.0 format (256x256)
            image = image.resize((256, 256))
            image_array = np.array(image) / 255.0  # Normalize to [0,1]
            
            # Convert to tensor [1, 256, 256]
            image_tensor = torch.FloatTensor(image_array).unsqueeze(0)
            
            # Apply transforms if provided
            if self.transform:
                image_tensor = self.transform(image_tensor)
            
            # Extract labels for 14 conditions
            labels = torch.FloatTensor([
                row[condition] for condition in self.conditions
            ])
            
            # Metadata (convert all to strings to avoid collate issues)
            metadata = {
                'Image_name': str(row['Image_name']),
                'Patient_ID': str(row['Patient_ID']),
                'Study': str(row['Study']),
                'Sex': str(row['Sex']),
                'Age': str(row['Age'] if pd.notna(row['Age']) else 'unknown'),
                'ViewCategory': str(row['ViewCategory']),
                'ViewPosition': str(row['ViewPosition'])
            }
            
            return {
                'image': image_tensor,
                'labels': labels,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return dummy data if image fails to load
            return {
                'image': torch.zeros(1, 256, 256),
                'labels': torch.zeros(14),
                'metadata': {'Image_name': 'ERROR', 'Patient_ID': 'ERROR'}
            }

def create_dataset_loaders(batch_size=4, max_samples=None):
    """
    Create dataset loaders for Grand X-Ray Slam
    
    Args:
        batch_size: Batch size for training
        max_samples: Limit samples for testing
    
    Returns:
        train_loader: DataLoader for training
        dataset: Full dataset object
    """
    
    # Paths
    csv_path = r"D:\NEBULA_DIVISION_A\datasets\grand-xray-slam-division-a\train1.csv"
    images_path = r"D:\NEBULA_DIVISION_A\datasets\grand-xray-slam-division-a\train1"
    
    # Create dataset
    dataset = GrandXRayDataset(
        csv_file=csv_path,
        image_dir=images_path,
        max_samples=max_samples
    )
    
    # Create data loader (no workers to avoid collate issues)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Single-threaded to avoid metadata collate issues
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    logger.info(f"DataLoader created: batch_size={batch_size}, num_batches={len(train_loader)}")
    
    return train_loader, dataset

def test_single_batch():
    """
    Test NEBULA v4.0 with a single batch from the real dataset
    """
    logger.info("="*60)
    logger.info("NEBULA v4.0 - Grand X-Ray Slam Dataset Test")
    logger.info("="*60)
    
    # Create dataset loader (limit to 8 samples for quick test)
    train_loader, dataset = create_dataset_loaders(batch_size=2, max_samples=8)
    
    # Initialize NEBULA system
    config = NEBULAControlPanelV4()
    nebula_system = NEBULATrainerV4(config)
    
    logger.info(f"Testing with first batch from {len(dataset)} samples")
    
    # Process first batch
    for batch_idx, batch in enumerate(train_loader):
        logger.info(f"Processing batch {batch_idx + 1}/{len(train_loader)}")
        logger.info(f"Image shape: {batch['image'].shape}")
        logger.info(f"Labels shape: {batch['labels'].shape}")
        
        # Move to CUDA
        images = batch['image'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        
        logger.info(f"Sample metadata: {batch['metadata']['Image_name'][0]}")
        logger.info(f"Conditions detected: {labels[0].sum().item():.0f}/14")
        
        # Forward pass through NEBULA
        try:
            with torch.no_grad():
                outputs = nebula_system.model(images)
                logger.info(f"NEBULA output shape: {outputs.shape}")
                logger.info(f"Predictions: {torch.sigmoid(outputs[0]).cpu().numpy()[:5]} (first 5)")
                
            logger.info("✅ NEBULA v4.0 successfully processed real medical images!")
            
        except Exception as e:
            logger.error(f"Error in NEBULA forward pass: {e}")
            return False
        
        # Only test first batch
        break
    
    return True

def generate_competition_csv(model, dataset_loader, output_path):
    """
    Generate competition-compatible CSV file with checkpointing
    Format: Image_name,Atelectasis,Cardiomegaly,...,Support Devices
    """
    logger.info("Generating competition CSV file with checkpoint system...")
    
    results = []
    condition_names = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
        'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
        'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
        'Pneumothorax', 'Support Devices'
    ]
    
    checkpoint_dir = Path(r"C:\SANTO_GRIAL\checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        total_batches = len(dataset_loader)
        batches_per_epoch = max(1, total_batches // 10)  # Save every 10% progress
        
        for batch_idx, batch in enumerate(tqdm(dataset_loader, desc="Processing 107,374 images")):
            images = batch['image'].to(DEVICE)
            image_names = batch['metadata']['Image_name']
            
            # Get predictions from NEBULA
            outputs = model(images)
            predictions = torch.sigmoid(outputs).cpu().numpy()
            
            # Convert to binary predictions (0 or 1) - competition format requirement
            binary_predictions = (predictions > 0.5).astype(int)
            
            # Add to results
            for i, image_name in enumerate(image_names):
                result_row = {'Image_name': image_name}
                for j, condition in enumerate(condition_names):
                    result_row[condition] = binary_predictions[i][j]
                results.append(result_row)
            
            # Checkpoint every epoch (10% progress)
            if (batch_idx + 1) % batches_per_epoch == 0 or batch_idx == total_batches - 1:
                epoch_num = (batch_idx + 1) // batches_per_epoch
                progress = ((batch_idx + 1) / total_batches) * 100
                
                # Save checkpoint
                checkpoint_path = checkpoint_dir / f"nebula_checkpoint_epoch_{epoch_num:02d}.csv"
                checkpoint_df = pd.DataFrame(results)
                checkpoint_df.to_csv(checkpoint_path, index=False)
                
                logger.info(f"🔄 CHECKPOINT {epoch_num}/10: {progress:.1f}% complete - {len(results):,} images processed")
                logger.info(f"💾 Saved: {checkpoint_path}")
    
    # Save final CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    logger.info(f"✅ Final competition CSV saved: {output_path}")
    logger.info(f"📊 Total images processed: {len(results):,}")
    return output_path

def main():
    """
    Main execution: Process complete Grand X-Ray Slam dataset for competition
    """
    logger.info("🚀 NEBULA v4.0 Dataset Processing Started")
    logger.info("📊 Grand X-Ray Slam Division A - 107,374 Images")
    
    # Test with single batch first
    success = test_single_batch()
    
    if success:
        logger.info("✅ NEBULA v4.0 validated - Processing FULL dataset now")
        
        # Process complete dataset for competition
        logger.info("🏭 Creating production dataset loader for 107,374 images...")
        train_loader, dataset = create_dataset_loaders(batch_size=16, max_samples=None)
        
        # Initialize NEBULA system for production
        config = NEBULAControlPanelV4()
        nebula_system = NEBULATrainerV4(config)
        
        # Generate competition CSV
        output_path = r"C:\SANTO_GRIAL\nebula_competition_submission.csv"
        logger.info(f"🏆 Generating competition submission: {output_path}")
        
        generate_competition_csv(nebula_system.model, train_loader, output_path)
        
        logger.info("✅ Competition submission file generated successfully!")
        logger.info(f"📄 File: {output_path}")
        
    else:
        logger.error("❌ Issues detected. Please review before full processing.")
    
if __name__ == "__main__":
    main()