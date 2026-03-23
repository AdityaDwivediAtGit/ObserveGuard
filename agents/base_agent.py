"""
Base Agent Class for ObserveGuard Framework
Defines the interface for all agent implementations
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent execution states"""
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class Observation:
    """Represents an observation from the environment"""
    modality: str  # 'vision', 'audio', 'text', 'multimodal'
    data: Any
    timestamp: float
    quality_score: float = 1.0  # 0-1, for drift detection
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Action:
    """Represents an action to take in the environment"""
    action_type: str  # 'click', 'type', 'scroll', 'wait', 'stop'
    parameters: Dict[str, Any]
    confidence: float = 1.0
    reasoning: str = ""


@dataclass
class Thought:
    """Represents internal reasoning of the agent"""
    reasoning: str
    confidence: float
    next_action_type: str


@dataclass
class TrajectoryStep:
    """One step in agent trajectory: thought + observation + action"""
    thought: Optional[Thought]
    observation: Optional[Observation]
    action: Optional[Action]
    is_successful: bool = False


class BaseAgent(ABC):
    """
    Abstract base class for all agents in ObserveGuard framework.
    Implements the ReAct loop: Think -> Act -> Observe
    """

    def __init__(self, config: Dict[str, Any], agent_id: str = "base_agent"):
        """
        Initialize base agent
        
        Args:
            config: Configuration dict with model paths, hyperparameters
            agent_id: Unique identifier for this agent instance
        """
        self.config = config
        self.agent_id = agent_id
        self.state = AgentState.IDLE
        self.trajectory: List[TrajectoryStep] = []
        self.step_count = 0
        self.max_steps = config.get('max_steps', 20)
        logger.info(f"Initialized {self.__class__.__name__} with ID {agent_id}")

    @abstractmethod
    def think(self, observation: Optional[Observation]) -> Thought:
        """
        Reasoning step: Process observation and generate thought.
        
        Args:
            observation: Current environment observation or None at start
            
        Returns:
            Thought object with reasoning and next action type
        """
        pass

    @abstractmethod
    def act(self, thought: Thought) -> Action:
        """
        Action step: Convert thought to concrete action.
        
        Args:
            thought: Thought from reasoning step
            
        Returns:
            Action object to execute in environment
        """
        pass

    @abstractmethod
    def observe(self, action: Action) -> Observation:
        """
        Observation step: Execute action and observe result.
        
        Args:
            action: Action to execute
            
        Returns:
            Observation with environment state after action
        """
        pass

    def process_observation(self, observation: Observation) -> Observation:
        """
        Post-process observation (e.g., for quality assessment).
        Can be overridden by subclasses.
        
        Args:
            observation: Raw observation from environment
            
        Returns:
            Processed observation
        """
        return observation

    def step(self, last_observation: Optional[Observation] = None) -> TrajectoryStep:
        """
        Execute one ReAct loop: Think -> Act -> Observe
        
        Args:
            last_observation: Previous observation from environment
            
        Returns:
            TrajectoryStep with all components
        """
        self.state = AgentState.THINKING
        thought = self.think(last_observation)
        
        self.state = AgentState.ACTING
        action = self.act(thought)
        
        self.state = AgentState.OBSERVING
        observation = self.observe(action)
        observation = self.process_observation(observation)
        
        step = TrajectoryStep(
            thought=thought,
            observation=observation,
            action=action,
            is_successful=thought.confidence > self.config.get('success_threshold', 0.7)
        )
        
        self.trajectory.append(step)
        self.step_count += 1
        return step

    def run(self, task_description: str, max_steps: Optional[int] = None) -> Dict[str, Any]:
        """
        Run agent on a task until completion or max steps reached.
        
        Args:
            task_description: High-level task description
            max_steps: Override max steps if provided
            
        Returns:
            Dict with trajectory, success status, metrics
        """
        if max_steps:
            self.max_steps = max_steps
        
        logger.info(f"Starting task: {task_description}")
        self.trajectory = []
        self.step_count = 0
        self.state = AgentState.IDLE
        
        observation = None
        final_success = False
        
        try:
            while self.step_count < self.max_steps:
                step = self.step(observation)
                observation = step.observation
                
                # Check termination conditions
                if self._should_terminate(step):
                    final_success = True
                    break
                    
        except Exception as e:
            logger.error(f"Error during task execution: {e}")
            self.state = AgentState.ERROR
            final_success = False
        
        self.state = AgentState.COMPLETED if final_success else AgentState.ERROR
        
        return {
            'task': task_description,
            'success': final_success,
            'steps': self.step_count,
            'trajectory': [self._serialize_step(s) for s in self.trajectory],
            'final_state': self.state.value
        }

    def _should_terminate(self, step: TrajectoryStep) -> bool:
        """
        Determine if agent should stop. Override in subclasses for custom logic.
        
        Args:
            step: Latest trajectory step
            
        Returns:
            True if should terminate
        """
        if step.action.action_type == 'stop':
            return True
        return False

    def _serialize_step(self, step: TrajectoryStep) -> Dict[str, Any]:
        """Convert TrajectoryStep to JSON-serializable dict"""
        return {
            'thought': {
                'reasoning': step.thought.reasoning if step.thought else None,
                'confidence': step.thought.confidence if step.thought else None,
                'next_action': step.thought.next_action_type if step.thought else None,
            } if step.thought else None,
            'action': {
                'type': step.action.action_type,
                'parameters': step.action.parameters,
                'confidence': step.action.confidence,
                'reasoning': step.action.reasoning,
            } if step.action else None,
            'observation': {
                'modality': step.observation.modality,
                'timestamp': step.observation.timestamp,
                'quality': step.observation.quality_score,
            } if step.observation else None,
            'successful': step.is_successful,
        }

    def get_trajectory_summary(self) -> Dict[str, Any]:
        """Get summary statistics of current trajectory"""
        if not self.trajectory:
            return {}
        
        successful_steps = sum(1 for s in self.trajectory if s.is_successful)
        avg_confidence = sum(s.action.confidence for s in self.trajectory if s.action) / len(self.trajectory)
        
        return {
            'total_steps': len(self.trajectory),
            'successful_steps': successful_steps,
            'success_rate': successful_steps / len(self.trajectory),
            'avg_action_confidence': avg_confidence,
        }

    def save_trajectory(self, filepath: str):
        """Save trajectory to JSON file"""
        data = {
            'agent_id': self.agent_id,
            'trajectory': [self._serialize_step(s) for s in self.trajectory],
            'summary': self.get_trajectory_summary(),
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Trajectory saved to {filepath}")
