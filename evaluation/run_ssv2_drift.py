"""
SSv2 Drift Evaluation
Test agent robustness under distribution shift (noise, drift) scenarios
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import argparse
import numpy as np
from pathlib import Path

from agents import ReActAgent, ObserveGuard
from evaluation.metrics import MetricsCalculator, EvaluationResults
from datasets import prepare_ssv2_dataset

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class SSv2DriftEvaluator:
    """Evaluate agent robustness under multimodal distribution shift"""
    
    def __init__(self,
                 agent_type: str = 'react',
                 guard_enabled: bool = False,
                 output_dir: str = './results'):
        """
        Initialize SSv2 drift evaluator
        
        Args:
            agent_type: 'react' or 'observe_guard'
            guard_enabled: Whether to use ObserveGuard
            output_dir: Output directory
        """
        self.agent_type = agent_type
        self.guard_enabled = guard_enabled
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.metrics_calc = MetricsCalculator()
        self.results_by_noise = {}
        
        logger.info(f"Initialized SSv2DriftEvaluator: agent={agent_type}, guard={guard_enabled}")
    
    def setup_agent(self, config: Dict[str, Any]):
        """Create agent"""
        base_agent = ReActAgent(config, agent_id="react_agent_ssv2")
        
        if self.guard_enabled:
            self.agent = ObserveGuard(base_agent, config, agent_id="observe_guard_ssv2")
        else:
            self.agent = base_agent
    
    def run_drift_scenario(self,
                          noise_level: float,
                          num_videos: int = 10) -> Dict[str, Any]:
        """
        Run evaluation on videos with specific noise level
        
        Args:
            noise_level: Noise level (0.0-0.4)
            num_videos: Number of videos to process
            
        Returns:
            Results for this noise level
        """
        logger.info(f"Running SSv2 evaluation at noise level {noise_level}...")
        
        trajectories = []
        success_count = 0
        
        for vid_idx in range(num_videos):
            # Simulate video processing task
            task_desc = f"Analyze video with noise={noise_level}: video_{vid_idx:04d}"
            
            try:
                result = self.agent.run(task_desc, max_steps=15)
                trajectories.append(result)
                
                if result['success']:
                    success_count += 1
                    
            except Exception as e:
                logger.warning(f"Error processing video {vid_idx}: {e}")
        
        # Calculate metrics
        perf_metrics = self.metrics_calc.calculate_performance_metrics(trajectories)
        
        # Security metrics if guard enabled
        sec_metrics = None
        if self.guard_enabled and hasattr(self.agent, 'security_log'):
            security_logs = [
                {
                    'anomaly_score': m.anomaly_score,
                    'is_attack': m.is_attack_suspected,
                    'rebinding': m.rebinding_detected,
                    'drift': m.observation_drift,
                }
                for m in self.agent.security_log
            ]
            sec_metrics = self.metrics_calc.calculate_security_metrics(security_logs)
        
        return {
            'noise_level': noise_level,
            'num_videos': num_videos,
            'success_rate': perf_metrics.task_success_rate,
            'mean_steps': perf_metrics.mean_steps_per_task,
            'mean_confidence': perf_metrics.mean_action_confidence,
            'security_metrics': {
                'anomaly_score': sec_metrics.mean_anomaly_score if sec_metrics else None,
                'drift_detection': sec_metrics.attack_detection_rate if sec_metrics else None,
            } if sec_metrics else None,
        }
    
    def run_robustness_sweep(self,
                            noise_levels: List[float],
                            videos_per_level: int = 10) -> Dict[str, Any]:
        """
        Run robustness evaluation across noise levels
        
        Args:
            noise_levels: List of noise levels to test
            videos_per_level: Videos per noise level
            
        Returns:
            Dict with results for each noise level
        """
        logger.info(f"Starting robustness sweep: {len(noise_levels)} noise levels")
        
        all_results = []
        
        for noise_level in noise_levels:
            result = self.run_drift_scenario(noise_level, videos_per_level)
            all_results.append(result)
            self.results_by_noise[noise_level] = result
        
        return {
            'noise_levels': noise_levels,
            'results_by_level': all_results,
        }
    
    def analyze_drift_performance(self) -> Dict[str, Any]:
        """Analyze how performance degrades with noise"""
        if not self.results_by_noise:
            return {}
        
        noise_levels = sorted(self.results_by_noise.keys())
        success_rates = [self.results_by_noise[n]['success_rate'] for n in noise_levels]
        confidences = [self.results_by_noise[n]['mean_confidence'] for n in noise_levels]
        
        # Calculate degradation
        baseline_success = success_rates[0]
        degradation_rates = [
            (baseline_success - sr) / baseline_success * 100
            for sr in success_rates
        ]
        
        # Estimate robustness
        robustness_score = 1.0 - (max(degradation_rates) / 100)
        
        return {
            'noise_levels': noise_levels,
            'success_rates': success_rates,
            'degradation_rates': degradation_rates,
            'mean_degradation': np.mean(degradation_rates),
            'robustness_score': robustness_score,
            'confidences': confidences,
        }
    
    def save_results(self, sweep_results: Dict[str, Any], prefix: str = "ssv2"):
        """Save evaluation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Analyze drift performance
        analysis = self.analyze_drift_performance()
        
        # Full results dict
        results_dict = {
            'agent_type': self.agent_type,
            'guard_enabled': self.guard_enabled,
            'timestamp': timestamp,
            'sweep_results': sweep_results['results_by_level'],
            'analysis': analysis,
        }
        
        # Save JSON
        results_file = self.output_dir / f"{prefix}_drift_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"SSv2 Drift Evaluation Summary ({self.agent_type})")
        print(f"{'='*60}")
        print(f"\nNoise Level vs Success Rate:")
        for nl, sr in zip(analysis.get('noise_levels', []), analysis.get('success_rates', [])):
            print(f"  Noise {nl:.1%}: {sr:.1f}% success")
        print(f"\nRobustness Analysis:")
        print(f"  Mean Degradation: {analysis.get('mean_degradation', 0):.1f}%")
        print(f"  Robustness Score: {analysis.get('robustness_score', 0):.3f}")
        print(f"{'='*60}\n")
    
    def create_drift_curves(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Create drift performance curves"""
        analysis = self.analyze_drift_performance()
        
        curve_data = {
            'noise_levels': analysis.get('noise_levels', []),
            'success_rates': analysis.get('success_rates', []),
            'degradation_rates': analysis.get('degradation_rates', []),
            'confidences': analysis.get('confidences', []),
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(curve_data, f, indent=2)
        
        return curve_data


def main():
    """Main SSv2 drift evaluation script"""
    parser = argparse.ArgumentParser(description='Evaluate agent robustness on noisy SSv2')
    parser.add_argument('--agent', type=str, choices=['react', 'observe_guard'],
                       default='observe_guard', help='Agent type')
    parser.add_argument('--noise-levels', type=float, nargs='+',
                       default=[0.0, 0.1, 0.2, 0.3],
                       help='Noise levels to test')
    parser.add_argument('--videos-per-level', type=int, default=10,
                       help='Videos to process per noise level')
    parser.add_argument('--output', type=str, default='./results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Prepare dataset
    logger.info("Preparing SSv2 dataset...")
    dataset = prepare_ssv2_dataset(
        output_dir="./data/ssv2",
        noise_levels=args.noise_levels
    )
    
    # Configure agent
    config = {
        'max_steps': 15,
        'reasoning_confidence_threshold': 0.6,
        'model_name': 'gpt-3.5-turbo',
        'probe_count': 3,
        'anomaly_threshold': 0.75,
        'tau': 0.85,
        'enable_probes': args.agent == 'observe_guard',
    }
    
    # Create evaluator
    evaluator = SSv2DriftEvaluator(
        agent_type=args.agent,
        guard_enabled=args.agent == 'observe_guard',
        output_dir=args.output
    )
    
    evaluator.setup_agent(config)
    
    # Run robustness sweep
    sweep_results = evaluator.run_robustness_sweep(
        args.noise_levels,
        args.videos_per_level
    )
    
    # Save results
    evaluator.save_results(sweep_results)


if __name__ == "__main__":
    main()
