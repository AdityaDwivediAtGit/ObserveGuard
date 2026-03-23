"""
Something-Something-v2 Dataset Augmentation
Adds distribution shift, noise, and non-IID sequencing for robustness testing
"""

import os
import json
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    """Configuration for augmentation parameters"""
    gaussian_noise_std: float = 0.2  # Gaussian noise std dev
    audio_perturbation_db: float = 10.0  # Audio noise level in dB
    frame_dropout_rate: float = 0.15  # Probability of dropping frames
    temporal_jitter: float = 0.1  # Temporal shift in seconds
    color_shift_magnitude: float = 0.2  # Color jitter magnitude
    blur_kernel_size: int = 5  # Gaussian blur kernel
    brightness_delta: float = 0.3  # Brightness adjustment


class SSv2Augmentor:
    """Augment SSv2 videos with different types of distribution shift"""
    
    def __init__(self, 
                 data_dir: str = "./data/ssv2",
                 output_dir: Optional[str] = None,
                 config: Optional[AugmentationConfig] = None):
        """
        Initialize SSv2 augmentor
        
        Args:
            data_dir: Directory with original SSv2 data
            output_dir: Output directory for augmented data
            config: Augmentation configuration
        """
        self.data_dir = data_dir
        self.output_dir = output_dir or os.path.join(data_dir, 'augmented')
        self.config = config or AugmentationConfig()
        
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Initialized SSv2 augmentor: {data_dir} -> {self.output_dir}")
    
    def add_gaussian_noise(self, 
                          video_data: np.ndarray,
                          noise_level: float) -> np.ndarray:
        """
        Add Gaussian noise to video frames.
        
        Args:
            video_data: Video array [frames, height, width, channels]
            noise_level: Noise standard deviation (0-1)
            
        Returns:
            Noisy video
        """
        noise = np.random.normal(
            0, 
            noise_level * self.config.gaussian_noise_std, 
            video_data.shape
        )
        noisy_video = np.clip(video_data + noise, 0, 1)
        return noisy_video
    
    def add_audio_perturbation(self,
                              audio_data: np.ndarray,
                              snr_db: float) -> np.ndarray:
        """
        Add audio noise to reduce SNR (signal-to-noise ratio).
        
        Args:
            audio_data: Audio signal
            snr_db: Target SNR in dB
            
        Returns:
            Perturbed audio
        """
        signal_power = np.mean(audio_data ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        
        noise = np.random.normal(0, np.sqrt(noise_power), audio_data.shape)
        perturbed = audio_data + noise
        
        return np.clip(perturbed, -1, 1)
    
    def add_temporal_jitter(self,
                           video_data: np.ndarray,
                           jitter_std: float) -> np.ndarray:
        """
        Add temporal jitter by randomly skipping/reordering frames.
        
        Args:
            video_data: Video array
            jitter_std: Jitter standard deviation in frames
            
        Returns:
            Jittered video
        """
        num_frames = video_data.shape[0]
        
        # Random frame shifts
        shifts = np.random.normal(0, jitter_std, num_frames).astype(int)
        shifts = np.clip(shifts, -(num_frames-1), num_frames-1)
        
        # Create new frame order
        frame_indices = np.arange(num_frames)
        frame_indices = np.clip(frame_indices + shifts, 0, num_frames-1)
        
        jittered = video_data[frame_indices]
        return jittered
    
    def add_frame_dropout(self,
                         video_data: np.ndarray,
                         dropout_rate: float) -> np.ndarray:
        """
        Randomly drop frames and interpolate.
        
        Args:
            video_data: Video array
            dropout_rate: Probability of dropping each frame
            
        Returns:
            Video with dropped frames
        """
        num_frames = video_data.shape[0]
        dropout_mask = np.random.binomial(1, 1 - dropout_rate, num_frames)
        
        kept_indices = np.where(dropout_mask)[0]
        if len(kept_indices) < 2:
            return video_data
        
        # Interpolate missing frames
        dropped_video = video_data[kept_indices]
        
        # Simple repetition interpolation
        result = []
        for i in range(len(kept_indices) - 1):
            result.append(dropped_video[i])
            result.append((dropped_video[i] + dropped_video[i+1]) / 2)
        result.append(dropped_video[-1])
        
        return np.array(result)
    
    def add_color_shift(self,
                       video_data: np.ndarray,
                       shift_magnitude: float) -> np.ndarray:
        """
        Add color jitter and brightness shifts.
        
        Args:
            video_data: Video array (normalized to 0-1)
            shift_magnitude: Magnitude of color shifts
            
        Returns:
            Color-shifted video
        """
        # Random brightness adjustment
        brightness_factor = 1.0 + np.random.uniform(
            -self.config.brightness_delta,
            self.config.brightness_delta
        )
        
        # Random channel shifts
        channel_shifts = np.random.uniform(
            -shift_magnitude,
            shift_magnitude,
            size=3
        )
        
        shifted = video_data * brightness_factor
        if video_data.shape[-1] == 3:
            shifted = shifted + channel_shifts
        
        return np.clip(shifted, 0, 1)
    
    def create_non_iid_split(self, 
                            videos: List[str],
                            num_clients: int = 5,
                            alpha: float = 0.5,
                            seed: int = 42) -> Dict[str, List[str]]:
        """
        Create non-IID (non-independent and identically distributed) data splits.
        Simulates realistic edge scenarios where different devices see different
        distributions of data.
        
        Args:
            videos: List of video IDs
            num_clients: Number of clients (edge devices)
            alpha: Dirichlet concentration parameter (lower = more non-IID)
            seed: Random seed
            
        Returns:
            Dict mapping client ID to list of video IDs
        """
        np.random.seed(seed)
        
        # Get class distribution (simulate by video ID hash)
        num_classes = min(157, len(videos))  # SSv2 has 157 action classes
        class_assignments = np.array([hash(vid) % num_classes for vid in videos])
        
        # Use Dirichlet distribution for non-IID sampling
        client_split = {f'client_{i}': [] for i in range(num_clients)}
        
        for class_id in range(num_classes):
            class_videos = [videos[j] for j in range(len(videos)) 
                          if class_assignments[j] == class_id]
            
            # Dirichlet sample for this class
            proportions = np.random.dirichlet(
                np.ones(num_clients) * alpha
            )
            
            class_indices = np.random.permutation(len(class_videos))
            start = 0
            
            for client_id in range(num_clients):
                end = start + int(proportions[client_id] * len(class_videos))
                client_videos = class_videos[class_indices[start:end]]
                client_split[f'client_{client_id}'].extend(client_videos)
                start = end
        
        logger.info(f"Created non-IID split for {num_clients} clients (alpha={alpha})")
        return client_split
    
    def augment_dataset(self,
                       noise_levels: List[float] = [0.1, 0.2, 0.3],
                       num_videos_per_level: int = 10) -> Dict[str, Any]:
        """
        Create augmented versions of SSv2 with varying noise levels.
        
        Args:
            noise_levels: List of noise levels to create
            num_videos_per_level: How many videos to augment per level
            
        Returns:
            Dict with augmentation info
        """
        logger.info(f"Augmenting SSv2 with {len(noise_levels)} noise levels...")
        
        # In production, would load actual videos
        # For now, create metadata for augmented versions
        mock_videos = [f"video_{i:04d}" for i in range(num_videos_per_level * len(noise_levels))]
        
        augmented_info = {
            'original_count': 0,
            'augmented_count': 0,
            'noise_levels': noise_levels,
            'variants_by_level': {},
        }
        
        for noise_level in noise_levels:
            # Create variant for this noise level
            variant_name = f"ssv2_noise_{noise_level:.1f}"
            variant_videos = [f"{v}_noise{noise_level:.1f}" 
                            for v in mock_videos[:num_videos_per_level]]
            
            augmented_info['variants_by_level'][variant_name] = {
                'noise_level': noise_level,
                'num_videos': len(variant_videos),
                'videos': variant_videos,
            }
            augmented_info['augmented_count'] += len(variant_videos)
        
        # Save augmentation metadata
        metadata_path = os.path.join(self.output_dir, 'augmentation_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(augmented_info, f, indent=2)
        
        logger.info(f"Created {augmented_info['augmented_count']} augmented video variants")
        return augmented_info
    
    def create_drift_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """
        Create different distribution shift scenarios for robustness testing.
        
        Returns:
            Dict mapping scenario names to their configurations
        """
        scenarios = {
            'gradual_blur': {
                'description': 'Gradually increasing blur (compression artifacts)',
                'augmentations': ['blur'],
                'magnitude_range': [1, 3, 5, 7, 9],
            },
            'progressive_noise': {
                'description': 'Increasing Gaussian noise',
                'augmentations': ['gaussian_noise'],
                'magnitude_range': [0.05, 0.1, 0.15, 0.2, 0.25],
            },
            'temporal_corruption': {
                'description': 'Frame drops and temporal jitter',
                'augmentations': ['frame_dropout', 'temporal_jitter'],
                'magnitude_range': [0.1, 0.2, 0.3, 0.4, 0.5],
            },
            'audio_degradation': {
                'description': 'Decreasing audio SNR',
                'augmentations': ['audio_perturbation'],
                'magnitude_range': [30, 20, 10, 5, 0],
            },
            'combined_shift': {
                'description': 'Combined vision + audio shift',
                'augmentations': ['gaussian_noise', 'frame_dropout', 'audio_perturbation'],
                'magnitude_range': [0.2, 0.3, 0.4],
            },
        }
        
        # Save scenario definitions
        scenarios_path = os.path.join(self.output_dir, 'drift_scenarios.json')
        with open(scenarios_path, 'w') as f:
            json.dump(scenarios, f, indent=2)
        
        logger.info(f"Defined {len(scenarios)} distribution shift scenarios")
        return scenarios


def prepare_ssv2_dataset(
    output_dir: str = "./data/ssv2",
    noise_levels: List[float] = [0.1, 0.2, 0.3],
    create_non_iid: bool = True,
    create_drift_scenarios: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to prepare augmented SSv2 dataset.
    
    Args:
        output_dir: Output directory
        noise_levels: Noise levels for augmentation
        create_non_iid: Create non-IID splits
        create_drift_scenarios: Create drift scenario definitions
        
    Returns:
        Dict with preparation metadata
    """
    augmentor = SSv2Augmentor(output_dir=output_dir)
    
    # Augment dataset
    aug_info = augmentor.augment_dataset(
        noise_levels=noise_levels,
        num_videos_per_level=10
    )
    
    # Create non-IID splits
    non_iid_split = None
    if create_non_iid:
        mock_videos = [f"video_{i:04d}" for i in range(100)]
        non_iid_split = augmentor.create_non_iid_split(
            mock_videos,
            num_clients=5,
            alpha=0.5
        )
    
    # Create drift scenarios
    drift_scenarios = None
    if create_drift_scenarios:
        drift_scenarios = augmentor.create_drift_scenarios()
    
    return {
        'output_dir': output_dir,
        'augmentation_info': aug_info,
        'non_iid_split': non_iid_split,
        'drift_scenarios': drift_scenarios,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Prepare augmented SSv2
    result = prepare_ssv2_dataset(
        output_dir="./data/ssv2",
        noise_levels=[0.1, 0.2, 0.3],
        create_non_iid=True,
        create_drift_scenarios=True
    )
    
    print(f"Dataset prepared in: {result['output_dir']}")
    print(f"Augmented variants: {result['augmentation_info']['augmented_count']}")
    if result['non_iid_split']:
        print(f"Non-IID clients: {len(result['non_iid_split'])}")
    if result['drift_scenarios']:
        print(f"Drift scenarios: {len(result['drift_scenarios'])}")
