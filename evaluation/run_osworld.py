"""
OSWorld Benchmark Evaluation
Run agent evaluation on OSWorld tasks with performance and security tracking
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import argparse
from pathlib import Path

from agents import ReActAgent, ObserveGuard
from evaluation.metrics import MetricsCalculator, EvaluationResults, PerformanceMetrics, SecurityMetrics, EnergyMetrics
from datasets import download_osworld

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class OSWorldEvaluator:
    """Evaluate agents on OSWorld benchmark"""
    
    def __init__(self,
                 agent_type: str = 'react',
                 guard_enabled: bool = False,
                 output_dir: str = './results'):
        """
        Initialize OSWorld evaluator
        
        Args:
            agent_type: 'react' or 'observe_guard'
            guard_enabled: Whether to wrap with ObserveGuard
            output_dir: Output directory for results
        """
        self.agent_type = agent_type
        self.guard_enabled = guard_enabled
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.metrics_calc = MetricsCalculator()
        self.trajectories = []
        self.energy_log = {
            'total_kwh': 0.0,
            'co2_kg': 0.0,
            'memory_mb': 0.0,
        }
        
        logger.info(f"Initialized OSWorldEvaluator: agent={agent_type}, guard={guard_enabled}")
    
    def setup_agent(self, config: Dict[str, Any]):
        """Create and configure agent"""
        base_agent = ReActAgent(config, agent_id="react_agent_osworld")
        
        if self.guard_enabled:
            self.agent = ObserveGuard(base_agent, config, agent_id="observe_guard_osworld")
        else:
            self.agent = base_agent
        
        logger.info(f"Agent setup complete: {self.agent.agent_id}")
    
    def run_task(self, task_id: str, task_config: Dict[str, Any], 
                track_energy: bool = False) -> Dict[str, Any]:
        """
        Run a single task with the agent
        
        Args:
            task_id: Task identifier
            task_config: Task configuration
            track_energy: Whether to track energy during execution
            
        Returns:
            Dict with task results
        """
        logger.info(f"Running task {task_id}...")
        
        task_description = task_config.get('description', f'Task {task_id}')
        
        try:
            # Run agent on task
            results = self.agent.run(task_description, max_steps=task_config.get('max_steps', 20))
            
            # Track energy (simulated)
            if track_energy:
                energy_data = self._simulate_energy_tracking(results['steps'])
                self.energy_log['total_kwh'] += energy_data['kwh']
                self.energy_log['co2_kg'] += energy_data['co2']
            
            # Store trajectory
            self.trajectories.append(results)
            
            # Extract results
            return {
                'task_id': task_id,
                'success': results['success'],
                'steps': results['steps'],
                'trajectory': results['trajectory'],
                'energy_kwh': self.energy_log.get('kwh', 0.0),
            }
        
        except Exception as e:
            logger.error(f"Error running task {task_id}: {e}")
            return {
                'task_id': task_id,
                'success': False,
                'steps': 0,
                'error': str(e),
            }
    
    def evaluate_on_split(self,
                         task_list: List[str],
                         task_configs: Dict[str, Dict],
                         track_energy: bool = False) -> Dict[str, Any]:
        """
        Run evaluation on a set of tasks
        
        Args:
            task_list: List of task IDs to evaluate
            task_configs: Dict mapping task IDs to configs
            track_energy: Track energy during execution
            
        Returns:
            Evaluation results
        """
        logger.info(f"Starting evaluation on {len(task_list)} tasks...")
        
        results = []
        for task_id in task_list:
            if task_id in task_configs:
                result = self.run_task(task_id, task_configs[task_id], track_energy)
                results.append(result)
        
        logger.info(f"Completed evaluation on {len(results)} tasks")
        return results
    
    def compute_metrics(self) -> EvaluationResults:
        """Compute aggregated metrics"""
        perf_metrics = self.metrics_calc.calculate_performance_metrics(self.trajectories)
        
        # Extract security logs if guard is enabled
        security_logs = []
        if self.guard_enabled and hasattr(self.agent, 'security_log'):
            security_logs = [
                {
                    'anomaly_score': m.anomaly_score,
                    'is_attack': m.is_attack_suspected,
                    'rebinding': m.rebinding_detected,
                }
                for m in self.agent.security_log
            ]
        
        sec_metrics = self.metrics_calc.calculate_security_metrics(security_logs)
        
        # Energy metrics
        total_time = sum(len(t.get('trajectory', [])) for t in self.trajectories)
        eng_metrics = self.metrics_calc.calculate_energy_metrics(
            self.energy_log,
            num_tasks=len(self.trajectories),
            total_time_seconds=total_time
        )
        
        return EvaluationResults(
            task_name='osworld',
            agent_type=self.agent_type,
            performance=perf_metrics,
            security=sec_metrics,
            energy=eng_metrics,
            timestamp=datetime.now().isoformat(),
        )
    
    def save_results(self, results: EvaluationResults, prefix: str = "osworld"):
        """Save evaluation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        results_file = self.output_dir / f"{prefix}_results_{timestamp}.json"
        
        results_dict = {
            'task': results.task_name,
            'agent': results.agent_type,
            'timestamp': results.timestamp,
            'performance': {
                'success_rate': results.performance.task_success_rate,
                'mean_steps': results.performance.mean_steps_per_task,
                'mean_confidence': results.performance.mean_action_confidence,
                'completion_rate': results.performance.completion_rate,
                'error_recovery': results.performance.error_recovery_rate,
            },
            'security': {
                'detection_rate': results.security.attack_detection_rate,
                'false_positive': results.security.false_positive_rate,
                'rebinding_accuracy': results.security.rebinding_detection_accuracy,
                'mean_anomaly': results.security.mean_anomaly_score,
            },
            'energy': {
                'total_kwh': results.energy.total_energy_kwh,
                'co2_kg': results.energy.co2_emissions_kg,
                'inference_ms': results.energy.inference_time_ms,
                'energy_per_task': results.energy.energy_per_task,
                'edge_efficiency': results.energy.edge_device_efficiency,
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
        
        # Print summary
        print(f"\n{'='*50}")
        print(f"OSWorld Evaluation Summary ({self.agent_type})")
        print(f"{'='*50}")
        print(f"Success Rate: {results.performance.task_success_rate:.1f}%")
        print(f"Mean Steps: {results.performance.mean_steps_per_task:.2f}")
        print(f"Completion Rate: {results.performance.completion_rate:.1f}%")
        if self.guard_enabled:
            print(f"Attack Detection Rate: {results.security.attack_detection_rate:.1f}%")
            print(f"Rebinding Detection: {results.security.rebinding_detection_accuracy:.1f}%")
        print(f"Total Energy: {results.energy.total_energy_kwh:.6f} kWh")
        print(f"Edge Efficiency: {results.energy.edge_device_efficiency:.3f}")
        print(f"{'='*50}\n")
    
    def _simulate_energy_tracking(self, num_steps: int) -> Dict[str, float]:
        """Simulate energy tracking (would use codecarbon in production)"""
        # Rough estimates: Pi 5 @ ~5W, inference ~100ms per step
        power_w = 5.0
        time_per_step = 0.1
        
        energy_kwh = (power_w * num_steps * time_per_step) / (3.6e6)
        co2_kg = energy_kwh * 0.5  # Rough estimate: 500g CO2/kWh (India grid)
        
        return {
            'kwh': energy_kwh,
            'co2': co2_kg,
        }


def main():
    """Main evaluation script"""
    parser = argparse.ArgumentParser(description='Evaluate agents on OSWorld')
    parser.add_argument('--agent', type=str, choices=['react', 'observe_guard'],
                       default='observe_guard', help='Agent type')
    parser.add_argument('--tasks', type=str, default='verified',
                       choices=['verified', 'test', 'all'],
                       help='Task set to evaluate')
    parser.add_argument('--num-tasks', type=int, default=10,
                       help='Number of tasks to run')
    parser.add_argument('--output', type=str, default='./results',
                       help='Output directory')
    parser.add_argument('--track-energy', action='store_true',
                       help='Track energy during evaluation')
    
    args = parser.parse_args()
    
    # Download and prepare tasks
    logger.info("Preparing OSWorld dataset...")
    dataset = download_osworld(output_dir="./data/osworld")
    
    task_list = list(dataset['tasks'].keys())[:args.num_tasks]
    task_configs = dataset['tasks']
    
    # Configure agent
    config = {
        'max_steps': 20,
        'reasoning_confidence_threshold': 0.6,
        'model_name': 'gpt-3.5-turbo',
        'probe_count': 3,
        'anomaly_threshold': 0.75,
        'tau': 0.85,
        'enable_probes': args.agent == 'observe_guard',
    }
    
    # Create evaluator
    evaluator = OSWorldEvaluator(
        agent_type=args.agent,
        guard_enabled=args.agent == 'observe_guard',
        output_dir=args.output
    )
    
    evaluator.setup_agent(config)
    
    # Run evaluation
    results = evaluator.evaluate_on_split(
        task_list,
        task_configs,
        track_energy=args.track_energy
    )
    
    # Compute and save metrics
    eval_results = evaluator.compute_metrics()
    evaluator.save_results(eval_results, prefix=args.tasks)


if __name__ == "__main__":
    main()
