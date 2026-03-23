"""
ReAct (Reasoning + Acting) Agent Implementation
Extends BaseAgent with reasoning-focused behavior
"""

from typing import Any, Dict, List, Optional
import re
import logging
from agents.base_agent import BaseAgent, Observation, Action, Thought

logger = logging.getLogger(__name__)


class ReActAgent(BaseAgent):
    """
    ReAct agent that uses chain-of-thought reasoning for decision making.
    Implements the ReAct pattern: Thought -> Action -> Observation cycle
    """

    def __init__(self, config: Dict[str, Any], agent_id: str = "react_agent"):
        """
        Initialize ReAct agent
        
        Args:
            config: Configuration dict. Expects:
                - max_steps: max steps per episode
                - reasoning_confidence_threshold: min confidence to act
                - model_name: LLM to use for reasoning (optional)
            agent_id: Unique agent identifier
        """
        super().__init__(config, agent_id)
        self.reasoning_model = config.get('model_name', 'gpt-3.5-turbo')
        self.confidence_threshold = config.get('reasoning_confidence_threshold', 0.6)
        self.action_history: List[Dict[str, Any]] = []
        logger.info(f"Initialized ReActAgent with model: {self.reasoning_model}")

    def think(self, observation: Optional[Observation]) -> Thought:
        """
        Generate reasoning using chain-of-thought approach.
        
        Args:
            observation: Current observation from environment
            
        Returns:
            Thought with reasoning and next action type
        """
        # Build context from recent history
        context = self._build_context(observation)
        
        # Generate reasoning (in production, use actual LLM)
        reasoning = self._generate_reasoning(context)
        
        # Extract action type from reasoning
        next_action_type = self._extract_action_type(reasoning)
        
        # Estimate confidence based on reasoning clarity
        confidence = self._estimate_confidence(reasoning)
        
        thought = Thought(
            reasoning=reasoning,
            confidence=confidence,
            next_action_type=next_action_type
        )
        
        logger.debug(f"Generated thought with confidence {confidence:.2f}: {next_action_type}")
        return thought

    def act(self, thought: Thought) -> Action:
        """
        Convert thought to concrete action.
        
        Args:
            thought: Reasoning output from think() step
            
        Returns:
            Action with type and parameters
        """
        action_type = thought.next_action_type
        
        # Generate action parameters based on type
        parameters = self._generate_action_parameters(action_type)
        
        action = Action(
            action_type=action_type,
            parameters=parameters,
            confidence=thought.confidence,
            reasoning=thought.reasoning
        )
        
        self.action_history.append({
            'step': self.step_count,
            'action_type': action_type,
            'confidence': thought.confidence
        })
        
        return action

    def observe(self, action: Action) -> Observation:
        """
        Execute action and collect observation from environment.
        In a real system, this would interact with the actual environment.
        
        Args:
            action: Action to execute
            
        Returns:
            Observation from environment after action
        """
        import time
        
        # Simulate environment interaction
        observation_data = self._simulate_environment_step(action)
        
        observation = Observation(
            modality='multimodal',
            data=observation_data,
            timestamp=time.time(),
            quality_score=self._assess_observation_quality(observation_data),
            metadata={
                'action_type': action.action_type,
                'agent_id': self.agent_id,
            }
        )
        
        return observation

    def _build_context(self, observation: Optional[Observation]) -> Dict[str, Any]:
        """Build context from recent trajectory for reasoning"""
        context = {
            'step_count': self.step_count,
            'recent_actions': self.action_history[-3:] if self.action_history else [],
        }
        
        if observation:
            context['last_observation'] = {
                'modality': observation.modality,
                'quality': observation.quality_score,
                'timestamp': observation.timestamp,
            }
        
        return context

    def _generate_reasoning(self, context: Dict[str, Any]) -> str:
        """
        Generate chain-of-thought reasoning.
        In production, calls LLM. Here simplified simulation.
        
        Args:
            context: Context about current state
            
        Returns:
            Reasoning string
        """
        step = context['step_count']
        recent_actions = context.get('recent_actions', [])
        
        # Simulate reasoning
        reasoning_steps = []
        reasoning_steps.append(f"Step {step}: Analyzing current state.")
        
        if recent_actions:
            last_action = recent_actions[-1]
            reasoning_steps.append(f"Last action was {last_action['action_type']} "
                                 f"with confidence {last_action['confidence']:.2f}.")
        
        if 'last_observation' in context:
            obs_quality = context['last_observation']['quality']
            reasoning_steps.append(f"Observation quality: {obs_quality:.2f}")
        
        reasoning_steps.append("Determining next best action...")
        
        return " ".join(reasoning_steps)

    def _extract_action_type(self, reasoning: str) -> str:
        """
        Extract action type from reasoning.
        Uses heuristics or pattern matching.
        
        Args:
            reasoning: Reasoning string
            
        Returns:
            Action type string
        """
        # Simple heuristic-based extraction
        action_keywords = {
            'click': ['click', 'press', 'select'],
            'type': ['type', 'enter', 'input', 'write'],
            'scroll': ['scroll', 'down', 'up'],
            'wait': ['wait', 'pause', 'delay'],
            'stop': ['stop', 'done', 'complete'],
        }
        
        reasoning_lower = reasoning.lower()
        
        for action, keywords in action_keywords.items():
            for keyword in keywords:
                if keyword in reasoning_lower:
                    return action
        
        # Default to click if no pattern matches
        return 'click'

    def _estimate_confidence(self, reasoning: str) -> float:
        """
        Estimate confidence based on reasoning quality.
        
        Args:
            reasoning: Reasoning string
            
        Returns:
            Confidence score 0-1
        """
        # Simple heuristics
        length_score = min(len(reasoning) / 100, 1.0)  # Longer reasoning = more confident
        base_confidence = 0.7
        
        return min(base_confidence + 0.2 * length_score, 1.0)

    def _generate_action_parameters(self, action_type: str) -> Dict[str, Any]:
        """
        Generate parameters for the given action type.
        
        Args:
            action_type: Type of action to generate parameters for
            
        Returns:
            Dict of action parameters
        """
        parameters = {
            'action_type': action_type,
            'timestamp': self.step_count,
        }
        
        if action_type == 'click':
            parameters.update({
                'x': 500,
                'y': 500,
                'button': 'left',
            })
        elif action_type == 'type':
            parameters.update({
                'text': 'sample input',
            })
        elif action_type == 'scroll':
            parameters.update({
                'direction': 'down',
                'amount': 3,
            })
        elif action_type == 'wait':
            parameters.update({
                'duration': 1.0,
            })
        
        return parameters

    def _simulate_environment_step(self, action: Action) -> Dict[str, Any]:
        """
        Simulate environment response to action.
        In production, this would interact with real environment.
        
        Args:
            action: Action executed
            
        Returns:
            Simulated observation data
        """
        return {
            'action_executed': action.action_type,
            'success': True,
            'screen_state': f'state_after_{action.action_type}',
            'agent_step': self.step_count,
            'data': {
                'elements': self._generate_mock_elements(),
                'text_content': 'Sample screen text content',
            }
        }

    def _generate_mock_elements(self) -> List[Dict[str, Any]]:
        """Generate mock UI elements for simulated observations"""
        return [
            {'type': 'button', 'text': 'Next', 'x': 400, 'y': 400},
            {'type': 'input', 'placeholder': 'Enter text', 'x': 300, 'y': 300},
            {'type': 'text', 'content': 'Welcome', 'x': 200, 'y': 200},
        ]

    def _assess_observation_quality(self, observation_data: Dict[str, Any]) -> float:
        """
        Assess quality of observation for drift detection.
        
        Args:
            observation_data: Observation dictionary
            
        Returns:
            Quality score 0-1
        """
        # Simple heuristics
        quality = 1.0
        
        if not observation_data.get('action_executed'):
            quality -= 0.2
        
        if 'data' in observation_data and 'elements' in observation_data['data']:
            if len(observation_data['data']['elements']) == 0:
                quality -= 0.3
        
        return max(quality, 0.0)

    def _should_terminate(self, step) -> bool:
        """Override termination logic for ReAct agent"""
        # Terminate if action is 'stop'
        if step.action and step.action.action_type == 'stop':
            return True
        
        # Could add other termination criteria
        return False

    def reset(self):
        """Reset agent state for new episode"""
        self.trajectory = []
        self.step_count = 0
        self.action_history = []
        logger.info(f"ReActAgent {self.agent_id} reset")
