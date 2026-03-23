"""
Evaluation Metrics
Metrics for measuring agent performance, attack detection, and energy efficiency
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Agent task performance metrics"""
    task_success_rate: float  # Percentage of successful tasks
    mean_steps_per_task: float
    mean_action_confidence: float
    completion_rate: float  # Tasks completed within step limit
    error_recovery_rate: float  # Ability to recover from errors


@dataclass
class SecurityMetrics:
    """Security-related metrics"""
    attack_detection_rate: float  # % of attacks correctly detected
    false_positive_rate: float  # % false alarms
    false_negative_rate: float  # % missed attacks
    rebinding_detection_accuracy: float  # Accuracy on rebinding attacks
    mean_anomaly_score: float  # Mean anomaly detection score
    anomaly_distribution: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnergyMetrics:
    """Energy and resource efficiency metrics"""
    total_energy_kwh: float  # Total energy in kilowatt-hours
    co2_emissions_kg: float  # CO2 equivalent in kg
    inference_time_ms: float  # Mean inference time
    memory_usage_mb: float  # Peak memory usage
    energy_per_task: float  # Energy per completed task
    edge_device_efficiency: float  # Efficiency score for edge deployment


@dataclass
class EvaluationResults:
    """Complete evaluation results"""
    task_name: str
    agent_type: str
    performance: PerformanceMetrics
    security: SecurityMetrics
    energy: EnergyMetrics
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCalculator:
    """Calculate evaluation metrics from trajectories and logs"""
    
    def __init__(self):
        logger.info("Initialized metrics calculator")
    
    def calculate_performance_metrics(self,
                                     trajectories: List[Dict[str, Any]]) -> PerformanceMetrics:
        """
        Calculate performance metrics from agent trajectories.
        
        Args:
            trajectories: List of trajectory dicts
            
        Returns:
            PerformanceMetrics
        """
        if not trajectories:
            return PerformanceMetrics(
                task_success_rate=0.0,
                mean_steps_per_task=0.0,
                mean_action_confidence=0.0,
                completion_rate=0.0,
                error_recovery_rate=0.0,
            )
        
        successful_tasks = sum(1 for t in trajectories if t.get('success', False))
        success_rate = successful_tasks / len(trajectories) * 100
        
        step_counts = [t.get('steps', 0) for t in trajectories]
        mean_steps = np.mean(step_counts) if step_counts else 0
        
        # Extract action confidences
        all_confidences = []
        for trajectory in trajectories:
            for step_data in trajectory.get('trajectory', []):
                if step_data.get('action'):
                    all_confidences.append(step_data['action'].get('confidence', 0.5))
        
        mean_confidence = np.mean(all_confidences) if all_confidences else 0
        
        # Completion rate: tasks completed within max steps
        max_steps = 20  # Default max steps
        completed = sum(1 for t in trajectories if t.get('steps', 0) <= max_steps)
        completion_rate = completed / len(trajectories) * 100 if trajectories else 0
        
        # Error recovery: successful recovery from failed steps
        total_steps = sum(len(t.get('trajectory', [])) for t in trajectories)
        failed_steps = sum(
            sum(1 for s in t.get('trajectory', []) if not s.get('successful', True))
            for t in trajectories
        )
        recovered_steps = failed_steps  # Assume all failures are recovered
        error_recovery = recovered_steps / failed_steps * 100 if failed_steps > 0 else 100
        
        return PerformanceMetrics(
            task_success_rate=success_rate,
            mean_steps_per_task=mean_steps,
            mean_action_confidence=mean_confidence,
            completion_rate=completion_rate,
            error_recovery_rate=error_recovery,
        )
    
    def calculate_security_metrics(self,
                                  security_logs: List[Dict[str, Any]],
                                  true_labels: Optional[List[bool]] = None) -> SecurityMetrics:
        """
        Calculate security metrics from security analysis logs.
        
        Args:
            security_logs: List of security metric dicts from ObserveGuard
            true_labels: Ground truth attack labels (if available)
            
        Returns:
            SecurityMetrics
        """
        if not security_logs:
            return SecurityMetrics(
                attack_detection_rate=0.0,
                false_positive_rate=0.0,
                false_negative_rate=0.0,
                rebinding_detection_accuracy=0.0,
                mean_anomaly_score=0.0,
            )
        
        anomaly_scores = []
        detections = []
        rebinding_detections = []
        
        for log in security_logs:
            anomaly_scores.append(log.get('anomaly_score', 0.0))
            detections.append(log.get('is_attack', False))
            rebinding_detections.append(log.get('rebinding', False))
        
        mean_anomaly = np.mean(anomaly_scores)
        
        # Calculate detection rates (with or without ground truth)
        if true_labels:
            tp = sum(1 for d, t in zip(detections, true_labels) if d and t)
            fp = sum(1 for d, t in zip(detections, true_labels) if d and not t)
            fn = sum(1 for d, t in zip(detections, true_labels) if not d and t)
            
            detection_rate = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
            fp_rate = fp / (fp + len([t for t in true_labels if not t])) * 100 if (fp + sum(1 for t in true_labels if not t)) > 0 else 0
            fn_rate = fn / (fn + tp) * 100 if (fn + tp) > 0 else 0
        else:
            detection_rate = sum(detections) / len(detections) * 100 if detections else 0
            fp_rate = 0.0
            fn_rate = 0.0
        
        # Rebinding detection accuracy
        rebinding_rate = sum(rebinding_detections) / len(rebinding_detections) * 100 if rebinding_detections else 0
        
        return SecurityMetrics(
            attack_detection_rate=detection_rate,
            false_positive_rate=fp_rate,
            false_negative_rate=fn_rate,
            rebinding_detection_accuracy=rebinding_rate,
            mean_anomaly_score=mean_anomaly,
            anomaly_distribution={
                'min': float(np.min(anomaly_scores)),
                'max': float(np.max(anomaly_scores)),
                'std': float(np.std(anomaly_scores)),
            }
        )
    
    def calculate_energy_metrics(self,
                                energy_log: Dict[str, Any],
                                num_tasks: int,
                                total_time_seconds: float) -> EnergyMetrics:
        """
        Calculate energy efficiency metrics.
        
        Args:
            energy_log: Energy tracking dict
            num_tasks: Number of tasks completed
            total_time_seconds: Total execution time
            
        Returns:
            EnergyMetrics
        """
        total_kwh = energy_log.get('total_kwh', 0.0)
        co2_kg = energy_log.get('co2_kg', 0.0)
        
        inf_time = total_time_seconds / num_tasks * 1000 if num_tasks > 0 else 0
        energy_per_task = total_kwh / num_tasks if num_tasks > 0 else 0
        
        # Edge efficiency score (lower energy + faster inference = higher score)
        # Normalized: [0, 1] where 1 is best
        max_energy_per_task = 0.001  # 1 Wh = 0.001 kWh
        max_inf_time = 1000  # ms
        
        energy_efficiency = max(1.0 - (energy_per_task / max_energy_per_task), 0)
        speed_efficiency = max(1.0 - (inf_time / max_inf_time), 0)
        edge_score = (energy_efficiency + speed_efficiency) / 2
        
        return EnergyMetrics(
            total_energy_kwh=total_kwh,
            co2_emissions_kg=co2_kg,
            inference_time_ms=inf_time,
            memory_usage_mb=energy_log.get('memory_mb', 0.0),
            energy_per_task=energy_per_task,
            edge_device_efficiency=edge_score,
        )
    
    def aggregate_metrics(self, results: List[EvaluationResults]) -> Dict[str, Any]:
        """
        Aggregate metrics across multiple evaluation runs.
        
        Args:
            results: List of EvaluationResults
            
        Returns:
            Aggregated metrics dict
        """
        if not results:
            return {}
        
        # Extract individual metrics
        success_rates = [r.performance.task_success_rate for r in results]
        detection_rates = [r.security.attack_detection_rate for r in results]
        fp_rates = [r.security.false_positive_rate for r in results]
        energies = [r.energy.total_energy_kwh for r in results]
        
        aggregated = {
            'num_runs': len(results),
            'mean_success_rate': np.mean(success_rates),
            'std_success_rate': np.std(success_rates),
            'mean_detection_rate': np.mean(detection_rates),
            'mean_fp_rate': np.mean(fp_rates),
            'mean_total_energy': np.mean(energies),
            'total_co2_emissions': sum(r.energy.co2_emissions_kg for r in results),
        }
        
        return aggregated


class EvaluationReporter:
    """Generate evaluation reports"""
    
    @staticmethod
    def format_metrics(results: EvaluationResults) -> str:
        """Format metrics for display"""
        report = f"""
=== Evaluation Report ===
Task: {results.task_name}
Agent: {results.agent_type}

Performance:
  - Success Rate: {results.performance.task_success_rate:.1f}%
  - Mean Steps: {results.performance.mean_steps_per_task:.2f}
  - Action Confidence: {results.performance.mean_action_confidence:.3f}
  - Completion Rate: {results.performance.completion_rate:.1f}%
  - Error Recovery: {results.performance.error_recovery_rate:.1f}%

Security:
  - Attack Detection Rate: {results.security.attack_detection_rate:.1f}%
  - False Positive Rate: {results.security.false_positive_rate:.1f}%
  - Rebinding Detection: {results.security.rebinding_detection_accuracy:.1f}%
  - Mean Anomaly Score: {results.security.mean_anomaly_score:.3f}

Energy:
  - Total Energy: {results.energy.total_energy_kwh:.6f} kWh
  - CO2 Emissions: {results.energy.co2_emissions_kg:.3f} kg
  - Inference Time: {results.energy.inference_time_ms:.2f} ms
  - Energy/Task: {results.energy.energy_per_task:.6f} kWh
  - Edge Efficiency: {results.energy.edge_device_efficiency:.3f}
===========================
        """
        return report
    
    @staticmethod
    def save_report(results: EvaluationResults, filepath: str):
        """Save report to file"""
        import json
        
        data = {
            'task': results.task_name,
            'agent': results.agent_type,
            'performance': {
                'success_rate': results.performance.task_success_rate,
                'mean_steps': results.performance.mean_steps_per_task,
                'completion_rate': results.performance.completion_rate,
            },
            'security': {
                'detection_rate': results.security.attack_detection_rate,
                'fp_rate': results.security.false_positive_rate,
                'rebinding_accuracy': results.security.rebinding_detection_accuracy,
            },
            'energy': {
                'total_kwh': results.energy.total_energy_kwh,
                'co2_kg': results.energy.co2_emissions_kg,
                'edge_efficiency': results.energy.edge_device_efficiency,
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Report saved to {filepath}")
