"""
ObserveGuard Secure Agent Wrapper
Adds observation-centric security layer for anomaly detection and attack mitigation
"""

from typing import Any, Dict, List, Optional, Tuple
import logging
import numpy as np
from dataclasses import dataclass, field
from agents.base_agent import BaseAgent, Observation, Action, Thought, TrajectoryStep

logger = logging.getLogger(__name__)


@dataclass
class SecurityMetrics:
    """Metrics from security analysis"""
    anomaly_score: float  # 0-1, higher = more anomalous
    is_attack_suspected: bool
    attack_confidence: float  # 0-1
    rebinding_detected: bool
    observation_drift: float  # 0-1
    action_consistency: float  # 0-1
    details: Dict[str, Any] = field(default_factory=dict)


class TransitionModel:
    """
    Lightweight transition model g_theta that predicts expected observations
    given actions. Used for anomaly detection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize transition model"""
        self.config = config
        self.weights = self._initialize_weights()
        logger.info("Initialized transition model")
    
    def _initialize_weights(self) -> np.ndarray:
        """Initialize model weights (in practice, load from checkpoint)"""
        # Simple random initialization for demo
        return np.random.randn(10, 10)
    
    def predict(self, action: Action, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict expected observation given action.
        
        Args:
            action: Action to execute
            context: Context dict with state info
            
        Returns:
            Expected observation dict
        """
        # Simplified prediction: mock implementation
        prediction = {
            'expected_modality': 'visual',
            'expected_elements_count': self._estimate_element_count(action),
            'expected_confidence': 0.85,
        }
        return prediction
    
    def _estimate_element_count(self, action: Action) -> int:
        """Estimate number of UI elements after action"""
        base_count = 5
        if action.action_type == 'scroll':
            return base_count + 2
        elif action.action_type == 'click':
            return base_count + 1
        return base_count


class ProbeGenerator:
    """
    Generates security probes (test actions) to verify observation authenticity.
    Implements the probe-based verification from ObserveGuard paper.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize probe generator"""
        self.config = config
        self.probe_count = config.get('probe_count', 3)
        self.probe_types = ['noop', 'verify_state', 'compare_observation']
    
    def generate_probe(self, action: Action, observation: Observation) -> Action:
        """
        Generate a probe action to verify observation authenticity.
        
        Args:
            action: Original action taken
            observation: Observation received
            
        Returns:
            Probe action to execute
        """
        probe_type = self.probe_types[len(self._probe_queue) % len(self.probe_types)]
        
        probe = Action(
            action_type=f'probe_{probe_type}',
            parameters={
                'original_action': action.action_type,
                'observation_id': id(observation),
                'probe_type': probe_type,
            },
            confidence=0.9,
            reasoning=f"Security probe to verify {observation.modality} observation"
        )
        
        return probe
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize probe generator"""
        self.config = config
        self.probe_count = config.get('probe_count', 3)
        self.probe_types = ['noop', 'verify_state', 'compare_observation']
        self._probe_queue: List[Action] = []


class ObserveGuard(BaseAgent):
    """
    ObserveGuard: Observation-centric security wrapper for multimodal agents.
    
    Key features:
    1. Observation monitoring and drift detection
    2. Action-observation consistency checking
    3. Rebinding attack detection
    4. Probe-based verification
    5. Energy-efficient threat detection
    
    Wraps any BaseAgent and adds security layer without modifying core logic.
    """
    
    def __init__(self, 
                 base_agent: BaseAgent,
                 config: Dict[str, Any],
                 agent_id: str = "observe_guard"):
        """
        Initialize ObserveGuard wrapper
        
        Args:
            base_agent: Agent to wrap (ReActAgent, etc.)
            config: Configuration dict with security parameters:
                - probe_count: number of probes per step (default 3)
                - anomaly_threshold: threshold for anomaly detection (default 0.75)
                - tau: decision threshold for attack (default 0.85)
                - enable_probes: whether to use probes (default True)
            agent_id: Unique identifier
        """
        super().__init__(config, agent_id)
        self.base_agent = base_agent
        self.transition_model = TransitionModel(config)
        self.probe_generator = ProbeGenerator(config)
        
        # Security parameters
        self.anomaly_threshold = config.get('anomaly_threshold', 0.75)
        self.attack_decision_threshold = config.get('tau', 0.85)
        self.enable_probes = config.get('enable_probes', True)
        
        # Tracking
        self.security_log: List[SecurityMetrics] = []
        self.attacks_detected = 0
        self.rebinding_detections = 0
        
        logger.info(f"Initialized ObserveGuard wrapper for {base_agent.agent_id}")

    def think(self, observation: Optional[Observation]) -> Thought:
        """
        Delegate thinking to base agent.
        
        Args:
            observation: Current observation
            
        Returns:
            Thought from base agent
        """
        return self.base_agent.think(observation)

    def act(self, thought: Thought) -> Action:
        """
        Get action from base agent.
        
        Args:
            thought: Reasoning from base agent
            
        Returns:
            Action from base agent
        """
        return self.base_agent.act(thought)

    def observe(self, action: Action) -> Observation:
        """
        Observe from base agent, then apply security analysis.
        
        Args:
            action: Action executed
            
        Returns:
            Original observation (not modified)
        """
        # Get observation from base agent
        observation = self.base_agent.observe(action)
        
        # Perform security checks
        self._perform_security_analysis(action, observation)
        
        return observation

    def _perform_security_analysis(self, action: Action, observation: Observation) -> SecurityMetrics:
        """
        Comprehensive security analysis on action-observation pair.
        
        Args:
            action: Action executed
            observation: Observation received
            
        Returns:
            SecurityMetrics with analysis results
        """
        metrics = SecurityMetrics(
            anomaly_score=0.0,
            is_attack_suspected=False,
            attack_confidence=0.0,
            rebinding_detected=False,
            observation_drift=0.0,
            action_consistency=1.0,
        )
        
        # 1. Check observation drift
        metrics.observation_drift = self._check_observation_drift(observation)
        
        # 2. Check action-observation consistency
        metrics.action_consistency = self._check_action_consistency(action, observation)
        
        # 3. Check for rebinding attacks
        metrics.rebinding_detected = self._detect_rebinding_attack(action, observation)
        if metrics.rebinding_detected:
            self.rebinding_detections += 1
        
        # 4. Combine signals into anomaly score
        metrics.anomaly_score = self._compute_anomaly_score(metrics)
        
        # 5. Generate security decision
        if self.enable_probes and metrics.anomaly_score > self.anomaly_threshold:
            metrics = self._run_probe_verification(action, observation, metrics)
        
        metrics.is_attack_suspected = metrics.anomaly_score > self.attack_decision_threshold
        metrics.attack_confidence = metrics.anomaly_score
        
        if metrics.is_attack_suspected:
            self.attacks_detected += 1
            logger.warning(f"ATTACK SUSPECTED: anomaly_score={metrics.anomaly_score:.3f}, "
                         f"rebinding={metrics.rebinding_detected}, "
                         f"drift={metrics.observation_drift:.3f}")
        
        self.security_log.append(metrics)
        return metrics

    def _check_observation_drift(self, observation: Observation) -> float:
        """
        Detect distribution shift in observations (noise, corruption, etc.).
        
        Args:
            observation: Observation to check
            
        Returns:
            Drift score 0-1 (0=clean, 1=highly drifted)
        """
        # Use observation quality score as proxy for drift
        drift = 1.0 - observation.quality_score
        
        # Could also use statistical checks on modality-specific features
        # For now, simple quality-based drift
        
        return drift

    def _check_action_consistency(self, action: Action, observation: Observation) -> float:
        """
        Check if observation is consistent with executed action.
        
        Args:
            action: Action that was executed
            observation: Observation received
            
        Returns:
            Consistency score 0-1 (1=highly consistent, 0=inconsistent)
        """
        # Predict expected observation
        expected = self.transition_model.predict(
            action,
            context={'action_type': action.action_type}
        )
        
        # Compare with actual observation
        expected_confidence = expected.get('expected_confidence', 0.8)
        actual_quality = observation.quality_score
        
        consistency = min(actual_quality / expected_confidence, 1.0)
        
        return consistency

    def _detect_rebinding_attack(self, action: Action, observation: Observation) -> bool:
        """
        Detect rebinding attacks where attacker substitutes observation with
        cached/controlled data.
        
        Args:
            action: Current action
            observation: Observation to check
            
        Returns:
            True if rebinding attack suspected
        """
        # Check for suspicious patterns
        
        # Pattern 1: Same observation returned for different actions
        if len(self.trajectory) > 1:
            prev_step = self.trajectory[-1]
            if prev_step.observation:
                # If observations are suspiciously identical despite different actions
                if (action.action_type != prev_step.action.action_type and
                    observation.data == prev_step.observation.data):
                    logger.warning("Detected suspicious observation reuse (rebinding)")
                    return True
        
        # Pattern 2: Observation timestamp mismatch
        if len(self.trajectory) > 0:
            prev_obs = self.trajectory[-1].observation
            if prev_obs:
                # Observations too close in time (execution faster than physically possible)
                time_diff = observation.timestamp - prev_obs.timestamp
                if time_diff < 0.01:  # Less than 10ms, suspicious for real interactions
                    logger.warning("Detected suspicious timing (rebinding indicator)")
                    return True
        
        return False

    def _compute_anomaly_score(self, metrics: SecurityMetrics) -> float:
        """
        Combine multiple security signals into unified anomaly score.
        
        Args:
            metrics: SecurityMetrics with individual checks
            
        Returns:
            Combined anomaly score 0-1
        """
        # Weighted combination of signals
        weights = {
            'drift': 0.3,
            'consistency': 0.3,
            'rebinding': 0.4,
        }
        
        # Invert consistency (high consistency = low anomaly)
        anomaly = (
            weights['drift'] * metrics.observation_drift +
            weights['consistency'] * (1.0 - metrics.action_consistency) +
            weights['rebinding'] * (1.0 if metrics.rebinding_detected else 0.0)
        )
        
        return anomaly

    def _run_probe_verification(self, 
                                action: Action, 
                                observation: Observation,
                                metrics: SecurityMetrics) -> SecurityMetrics:
        """
        Run probe-based verification when anomaly suspected.
        
        Args:
            action: Original action
            observation: Suspicious observation
            metrics: Existing metrics
            
        Returns:
            Updated metrics with probe results
        """
        logger.info("Running probe verification...")
        
        probe_confirmations = 0
        for i in range(self.config.get('probe_count', 3)):
            probe = self.probe_generator.generate_probe(action, observation)
            probe_obs = self.base_agent.observe(probe)
            
            # Check if probe observation is consistent
            if probe_obs.quality_score > 0.8 and probe_obs.data != observation.data:
                probe_confirmations += 1
        
        # If probes confirm anomaly, increase confidence
        if probe_confirmations >= 2:
            metrics.anomaly_score = min(metrics.anomaly_score + 0.1, 1.0)
            metrics.details['probe_confirmations'] = probe_confirmations
        
        return metrics

    def get_security_summary(self) -> Dict[str, Any]:
        """Get summary of security analysis"""
        if not self.security_log:
            return {}
        
        anomaly_scores = [m.anomaly_score for m in self.security_log]
        attack_detections = sum(1 for m in self.security_log if m.is_attack_suspected)
        
        return {
            'total_steps_analyzed': len(self.security_log),
            'attacks_detected': self.attacks_detected,
            'rebinding_detections': self.rebinding_detections,
            'attack_detection_rate': attack_detections / len(self.security_log) if self.security_log else 0,
            'mean_anomaly_score': np.mean(anomaly_scores),
            'max_anomaly_score': np.max(anomaly_scores),
            'min_anomaly_score': np.min(anomaly_scores),
        }

    def save_security_log(self, filepath: str):
        """Save security analysis log to file"""
        import json
        
        log_data = {
            'agent_id': self.agent_id,
            'wrapped_agent': self.base_agent.agent_id,
            'security_summary': self.get_security_summary(),
            'detailed_metrics': [
                {
                    'step': i,
                    'anomaly_score': m.anomaly_score,
                    'is_attack': m.is_attack_suspected,
                    'rebinding': m.rebinding_detected,
                    'drift': m.observation_drift,
                    'consistency': m.action_consistency,
                } for i, m in enumerate(self.security_log)
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2)
        logger.info(f"Security log saved to {filepath}")
