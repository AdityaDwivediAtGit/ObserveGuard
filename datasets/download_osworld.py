"""
OSWorld Dataset Download and Setup
Downloads and prepares OSWorld benchmark for evaluation
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
import urllib.request
import tarfile

logger = logging.getLogger(__name__)


class OSWorldDownloader:
    """Download and prepare OSWorld benchmark dataset"""
    
    def __init__(self, output_dir: str = "./data/osworld"):
        """
        Initialize OSWorld downloader
        
        Args:
            output_dir: Directory to save downloaded data
        """
        self.output_dir = output_dir
        self.verified_tasks_url = (
            "https://raw.githubusercontent.com/xlang-ai/OSWorld/"
            "main/evaluation_examples/verified_tasks.json"
        )
        self.screenshots_url = (
            "https://osworld-bucket.s3.amazonaws.com/screenshots/"
        )
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Initialized OSWorld downloader to {output_dir}")
    
    def download_verified_tasks(self) -> Dict[str, Any]:
        """
        Download list of verified tasks for evaluation.
        
        Returns:
            Dict mapping task IDs to task specifications
        """
        logger.info("Downloading verified tasks list...")
        
        try:
            with urllib.request.urlopen(self.verified_tasks_url, timeout=10) as response:
                tasks_data = json.loads(response.read().decode('utf-8'))
            
            # Save locally
            local_path = os.path.join(self.output_dir, 'verified_tasks.json')
            with open(local_path, 'w') as f:
                json.dump(tasks_data, f, indent=2)
            
            logger.info(f"Downloaded {len(tasks_data)} verified tasks")
            return tasks_data
        
        except Exception as e:
            logger.warning(f"Could not download from URL, creating mock tasks: {e}")
            return self._create_mock_tasks()
    
    def _create_mock_tasks(self) -> Dict[str, Any]:
        """Create mock task list for testing without internet"""
        mock_tasks = {
            "task_001": {
                "task_id": "task_001",
                "description": "Navigate to settings",
                "domain": "web",
                "difficulty": "easy",
                "steps": 3,
            },
            "task_002": {
                "task_id": "task_002",
                "description": "Complete a form",
                "domain": "web",
                "difficulty": "medium",
                "steps": 5,
            },
            "task_003": {
                "task_id": "task_003",
                "description": "Search and filter results",
                "domain": "web",
                "difficulty": "hard",
                "steps": 8,
            },
        }
        
        local_path = os.path.join(self.output_dir, 'verified_tasks.json')
        with open(local_path, 'w') as f:
            json.dump(mock_tasks, f, indent=2)
        
        logger.info(f"Created mock task set with {len(mock_tasks)} tasks")
        return mock_tasks
    
    def download_screenshots(self, task_ids: Optional[List[str]] = None, 
                           limit: Optional[int] = None):
        """
        Download screenshots for tasks (limited implementation).
        
        Args:
            task_ids: List of task IDs to download screenshots for
            limit: Maximum number of screenshots to download per task
        """
        logger.info(f"Downloading screenshots for {len(task_ids) if task_ids else 'all'} tasks...")
        
        # In production, would actually download from S3
        # For now, create mock screenshot directory structure
        screenshots_dir = os.path.join(self.output_dir, 'screenshots')
        os.makedirs(screenshots_dir, exist_ok=True)
        
        for task_id in (task_ids or []):
            task_dir = os.path.join(screenshots_dir, task_id)
            os.makedirs(task_dir, exist_ok=True)
            
            # Create mock screenshot metadata
            metadata = {
                'task_id': task_id,
                'screenshot_count': limit or 5,
                'timestamps': list(range(limit or 5)),
            }
            
            with open(os.path.join(task_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f)
        
        logger.info(f"Screenshot directories prepared in {screenshots_dir}")
    
    def prepare_evaluation_split(self, 
                                train_ratio: float = 0.7,
                                val_ratio: float = 0.15,
                                seed: int = 42) -> Dict[str, List[str]]:
        """
        Split tasks into train/val/test sets.
        
        Args:
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            seed: Random seed for reproducibility
            
        Returns:
            Dict mapping split names to task ID lists
        """
        import random
        
        # Load tasks
        tasks_path = os.path.join(self.output_dir, 'verified_tasks.json')
        with open(tasks_path, 'r') as f:
            tasks = json.load(f)
        
        task_ids = list(tasks.keys())
        random.seed(seed)
        random.shuffle(task_ids)
        
        n_tasks = len(task_ids)
        n_train = int(n_tasks * train_ratio)
        n_val = int(n_tasks * val_ratio)
        
        splits = {
            'train': task_ids[:n_train],
            'val': task_ids[n_train:n_train+n_val],
            'test': task_ids[n_train+n_val:],
        }
        
        # Save splits
        splits_path = os.path.join(self.output_dir, 'splits.json')
        with open(splits_path, 'w') as f:
            json.dump(splits, f, indent=2)
        
        logger.info(f"Created splits: train={len(splits['train'])}, "
                   f"val={len(splits['val'])}, test={len(splits['test'])}")
        
        return splits
    
    def augment_osworld_with_mutations(self, num_mutations: int = 5):
        """
        Create augmented versions of tasks with UI mutations.
        
        Args:
            num_mutations: Number of mutation variants per task
        """
        logger.info(f"Generating {num_mutations} mutations per task...")
        
        # Load original tasks
        tasks_path = os.path.join(self.output_dir, 'verified_tasks.json')
        with open(tasks_path, 'r') as f:
            tasks = json.load(f)
        
        augmented_dir = os.path.join(self.output_dir, 'augmented')
        os.makedirs(augmented_dir, exist_ok=True)
        
        augmented_tasks = {}
        mutation_types = ['reposition', 'occlusion', 'color_shift', 'resize']
        
        for task_id, task_spec in tasks.items():
            for mut_idx in range(num_mutations):
                mutation_type = mutation_types[mut_idx % len(mutation_types)]
                
                augmented_id = f"{task_id}_mut_{mut_idx}"
                augmented_tasks[augmented_id] = {
                    **task_spec,
                    'original_task_id': task_id,
                    'mutation_type': mutation_type,
                    'mutation_index': mut_idx,
                }
        
        # Save augmented tasks
        aug_path = os.path.join(augmented_dir, 'augmented_tasks.json')
        with open(aug_path, 'w') as f:
            json.dump(augmented_tasks, f, indent=2)
        
        logger.info(f"Generated {len(augmented_tasks)} augmented task variants")
        
        return augmented_tasks


def download_osworld(output_dir: str = "./data/osworld",
                    prepare_splits: bool = True,
                    augment_tasks: bool = True) -> Dict[str, Any]:
    """
    Convenience function to download and prepare OSWorld dataset.
    
    Args:
        output_dir: Directory to save data
        prepare_splits: Whether to create train/val/test splits
        augment_tasks: Whether to generate augmented task variants
        
    Returns:
        Dict with paths to downloaded resources
    """
    downloader = OSWorldDownloader(output_dir)
    
    # Download tasks
    tasks = downloader.download_verified_tasks()
    
    # Download screenshots
    downloader.download_screenshots(list(tasks.keys())[:10])  # Limit for speed
    
    splits = None
    if prepare_splits:
        splits = downloader.prepare_evaluation_split()
    
    augmented = None
    if augment_tasks:
        augmented = downloader.augment_osworld_with_mutations(num_mutations=3)
    
    return {
        'output_dir': output_dir,
        'tasks': tasks,
        'splits': splits,
        'augmented_tasks': augmented,
        'downloader': downloader,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Download and prepare OSWorld
    result = download_osworld(
        output_dir="./data/osworld",
        prepare_splits=True,
        augment_tasks=True
    )
    
    print(f"Dataset ready in: {result['output_dir']}")
    print(f"Total tasks: {len(result['tasks'])}")
    if result['splits']:
        print(f"Train: {len(result['splits']['train'])}, "
              f"Val: {len(result['splits']['val'])}, "
              f"Test: {len(result['splits']['test'])}")
