"""
Energy and Carbon Tracking Utility
Wrapper around CodeCarbon for tracking energy consumption and CO2 emissions
"""

import os
import logging
from typing import Dict, Any, Optional
from contextlib import contextmanager
from dataclasses import dataclass
import json
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class EnergyMeasurement:
    """Energy measurement result"""
    total_energy_kwh: float
    co2_emissions_kg: float
    duration_seconds: float
    power_draw_w: float
    country: str = "India"
    timestamp: str = ""


class EnergyTracker:
    """Wrapper for energy and carbon tracking"""
    
    def __init__(self, 
                 offline_mode: bool = True,
                 country_iso_code: str = "IN",
                 log_dir: str = "./logs"):
        """
        Initialize energy tracker
        
        Args:
            offline_mode: Run in offline mode (no cloud calls)
            country_iso_code: Country for carbon intensity estimation
            log_dir: Directory to save logs
        """
        self.offline_mode = offline_mode
        self.country_code = country_iso_code
        self.log_dir = log_dir
        
        os.makedirs(log_dir, exist_ok=True)
        
        # Try to import codecarbon, fall back to mock if unavailable
        try:
            from codecarbon import EmissionsTracker
            self.codecarbon_available = True
            self.EmissionsTracker = EmissionsTracker
        except ImportError:
            logger.warning("CodeCarbon not available, using mock tracker")
            self.codecarbon_available = False
        
        logger.info(f"Initialized EnergyTracker: offline={offline_mode}, country={country_iso_code}")
    
    @contextmanager
    def track_energy(self, task_name: str = "task"):
        """
        Context manager for tracking energy during code block execution.
        
        Example:
            with tracker.track_energy("inference"):
                # run inference
                ...
            # measurements available in tracker
        
        Args:
            task_name: Name of task being tracked
            
        Yields:
            Energy tracker context
        """
        start_time = datetime.now().isoformat()
        
        if self.codecarbon_available and not self.offline_mode:
            try:
                tracker = self.EmissionsTracker(
                    country_iso_code=self.country_code,
                    log_file_path=os.path.join(self.log_dir, f"{task_name}.log"),
                    offline=self.offline_mode
                )
                tracker.start()
                
                try:
                    yield tracker
                finally:
                    emissions = tracker.stop()
                    self._log_measurement(task_name, emissions, start_time)
            
            except Exception as e:
                logger.warning(f"CodeCarbon tracking failed: {e}, using mock")
                yield self._create_mock_tracker(task_name)
        else:
            # Use mock tracker
            yield self._create_mock_tracker(task_name)
    
    def _create_mock_tracker(self, task_name: str):
        """Create mock tracker when CodeCarbon unavailable"""
        class MockTracker:
            def __init__(self):
                self.emissions = 0.0
                self.energy_consumed = 0.0
                self.start_time = datetime.now()
            
            def start(self):
                return self
            
            def stop(self):
                # Simulate: 5W power draw for ~1 second = 1.4e-6 kWh
                duration = (datetime.now() - self.start_time).total_seconds()
                power_w = 5.0  # Raspberry Pi 5 typical
                self.energy_consumed = (power_w * duration) / 3.6e6  # Convert to kWh
                
                # CO2 intensity: ~500g per kWh (India grid average)
                self.emissions = self.energy_consumed * 0.5
                
                return self.emissions
        
        return MockTracker()
    
    def _log_measurement(self, task_name: str, emissions: float, start_time: str):
        """Log energy measurement"""
        measurement = {
            'task': task_name,
            'emissions_kg': emissions if emissions else 0.0,
            'start_time': start_time,
            'end_time': datetime.now().isoformat(),
        }
        
        log_file = os.path.join(self.log_dir, "energy_measurements.jsonl")
        with open(log_file, 'a') as f:
            f.write(json.dumps(measurement) + '\n')
        
        logger.info(f"Task '{task_name}': ~{emissions:.6f} kg CO2 emitted")
    
    def get_total_emissions(self) -> float:
        """Get total CO2 emissions from all tracked tasks"""
        log_file = os.path.join(self.log_dir, "energy_measurements.jsonl")
        
        if not os.path.exists(log_file):
            return 0.0
        
        total = 0.0
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    total += data.get('emissions_kg', 0)
                except:
                    pass
        
        return total
    
    def estimate_energy_requirements(self,
                                    num_tasks: int,
                                    steps_per_task: int,
                                    power_draw_w: float = 5.0,
                                    inference_time_per_step_ms: float = 100.0) -> Dict[str, float]:
        """
        Estimate energy requirements for a workload.
        
        Args:
            num_tasks: Number of tasks to run
            steps_per_task: Average steps per task
            power_draw_w: Power consumption in watts
            inference_time_per_step_ms: Inference time per step in milliseconds
            
        Returns:
            Dict with energy estimates
        """
        total_steps = num_tasks * steps_per_task
        total_time_hours = (total_steps * inference_time_per_step_ms / 1000) / 3600
        
        total_kwh = (power_draw_w * total_time_hours) / 1000
        co2_kg = total_kwh * 0.5  # 500g CO2/kWh for India grid
        
        return {
            'total_steps': total_steps,
            'total_time_hours': total_time_hours,
            'estimated_energy_kwh': total_kwh,
            'estimated_co2_kg': co2_kg,
            'power_draw_w': power_draw_w,
            'per_task_kwh': total_kwh / num_tasks if num_tasks > 0 else 0,
        }
    
    def compare_energy_scenarios(self, *scenarios: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare energy requirements across different scenarios.
        
        Args:
            *scenarios: Variable number of scenario dicts with keys:
                       'name', 'num_tasks', 'steps_per_task', 'power_w'
            
        Returns:
            Comparison dict
        """
        comparisons = []
        
        for scenario in scenarios:
            estimate = self.estimate_energy_requirements(
                scenario.get('num_tasks', 100),
                scenario.get('steps_per_task', 20),
                scenario.get('power_w', 5.0)
            )
            
            comparisons.append({
                'scenario': scenario.get('name', 'unnamed'),
                'energy_kwh': estimate['estimated_energy_kwh'],
                'co2_kg': estimate['estimated_co2_kg'],
                'cost_usd': estimate['estimated_energy_kwh'] * 0.12,  # ~$0.12/kWh India
            })
        
        return {
            'comparisons': comparisons,
            'most_efficient': min(comparisons, key=lambda x: x['energy_kwh'])['scenario'],
        }


class MockEnergyTracker:
    """Mock tracker for testing without CodeCarbon dependency"""
    
    def __init__(self):
        self.measurements = []
    
    @contextmanager
    def track_energy(self, task_name: str = "task"):
        """Mock context manager"""
        yield self
    
    def start(self):
        return self
    
    def stop(self):
        """Return mock emission value"""
        return 0.001  # kg CO2


def create_energy_tracker(config: Dict[str, Any]) -> EnergyTracker:
    """
    Factory function to create energy tracker from config.
    
    Args:
        config: Configuration dict with energy settings
        
    Returns:
        Configured EnergyTracker
    """
    energy_config = config.get('energy', {})
    
    return EnergyTracker(
        offline_mode=energy_config.get('codecarbon', {}).get('offline_mode', True),
        country_iso_code=energy_config.get('codecarbon', {}).get('country_iso_code', 'IN'),
        log_dir=energy_config.get('codecarbon', {}).get('log_dir', './logs')
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    tracker = EnergyTracker(offline_mode=True)
    
    # Estimate energy for different scenarios
    scenarios = [
        {
            'name': 'Baseline ReAct',
            'num_tasks': 100,
            'steps_per_task': 20,
            'power_w': 5.0,
        },
        {
            'name': 'ObserveGuard (with probes)',
            'num_tasks': 100,
            'steps_per_task': 25,  # Extra steps for probes
            'power_w': 5.0,
        },
        {
            'name': 'Jetson Orin Nano',
            'num_tasks': 100,
            'steps_per_task': 20,
            'power_w': 15.0,  # Higher power draw
        },
    ]
    
    comparison = tracker.compare_energy_scenarios(*scenarios)
    
    print("\nEnergy Consumption Comparison:")
    print(f"{'Scenario':<30} {'Energy (kWh)':<15} {'CO2 (kg)':<15} {'Cost (USD)':<15}")
    print("-" * 75)
    
    for comp in comparison['comparisons']:
        print(f"{comp['scenario']:<30} {comp['energy_kwh']:<15.6f} {comp['co2_kg']:<15.6f} {comp['cost_usd']:<15.2f}")
    
    print(f"\nMost Energy Efficient: {comparison['most_efficient']}")
