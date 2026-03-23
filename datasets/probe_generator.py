"""
Synthetic Probe Generator
Generates security probes for ObserveGuard anomaly detection and verification
"""

import os
import json
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ProbeType(Enum):
    """Types of security probes"""
    NO_OP = "noop"  # Do nothing, verify state unchanged
    STATE_VERIFY = "state_verify"  # Request state confirmation
    OBSERVATION_COMPARE = "observation_compare"  # Compare observations
    UI_CONSISTENCY = "ui_consistency"  # Check UI consistency
    TIMING_CHECK = "timing_check"  # Check timing constraints
    MULTIMODAL_SYNC = "multimodal_sync"  # Verify audio-visual sync


@dataclass
class Probe:
    """A security probe specification"""
    probe_id: str
    probe_type: ProbeType
    parameters: Dict[str, Any]
    expected_response: Dict[str, Any]
    confidence_threshold: float = 0.8


class ProbeGenerator:
    """Generate synthetic security probes for attack detection"""
    
    def __init__(self, output_dir: str = "./data/probes", seed: int = 42):
        """
        Initialize probe generator
        
        Args:
            output_dir: Directory to save generated probes
            seed: Random seed for reproducibility
        """
        self.output_dir = output_dir
        self.seed = seed
        np.random.seed(seed)
        
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Initialized probe generator to {output_dir}")
    
    def generate_noop_probes(self, num_probes: int = 5) -> List[Probe]:
        """
        Generate no-op probes that verify state doesn't change.
        Used to detect state caching attacks.
        
        Args:
            num_probes: Number of probes to generate
            
        Returns:
            List of Probe objects
        """
        probes = []
        
        for i in range(num_probes):
            probe = Probe(
                probe_id=f"noop_{i:03d}",
                probe_type=ProbeType.NO_OP,
                parameters={
                    'action': 'wait',
                    'duration': 0.5,
                    'expected_no_change': True,
                },
                expected_response={
                    'screen_state_same': True,
                    'element_count_same': True,
                    'text_content_same': True,
                },
                confidence_threshold=0.9,
            )
            probes.append(probe)
        
        logger.info(f"Generated {num_probes} no-op probes")
        return probes
    
    def generate_state_verification_probes(self, 
                                          num_probes: int = 5,
                                          ui_elements: Optional[List[Dict]] = None) -> List[Probe]:
        """
        Generate probes that request explicit state confirmation.
        Detects compromised observation channels.
        
        Args:
            num_probes: Number of probes
            ui_elements: Mock UI elements to verify
            
        Returns:
            List of Probe objects
        """
        if ui_elements is None:
            ui_elements = self._generate_mock_elements()
        
        probes = []
        
        for i in range(num_probes):
            # Select random UI element to verify
            element = ui_elements[i % len(ui_elements)]
            
            probe = Probe(
                probe_id=f"state_verify_{i:03d}",
                probe_type=ProbeType.STATE_VERIFY,
                parameters={
                    'verify_element': element['id'],
                    'element_type': element['type'],
                    'expected_location': element['location'],
                },
                expected_response={
                    'element_found': True,
                    'location_verified': True,
                    'type_verified': True,
                    'visible': True,
                },
                confidence_threshold=0.85,
            )
            probes.append(probe)
        
        logger.info(f"Generated {num_probes} state verification probes")
        return probes
    
    def generate_observation_comparison_probes(self,
                                              num_probes: int = 5) -> List[Probe]:
        """
        Generate probes that request multiple observations of same state.
        Detects observation substitution attacks.
        
        Args:
            num_probes: Number of probes
            
        Returns:
            List of Probe objects
        """
        probes = []
        
        for i in range(num_probes):
            delay_ms = 100 + i * 50  # Varying delays
            
            probe = Probe(
                probe_id=f"obs_compare_{i:03d}",
                probe_type=ProbeType.OBSERVATION_COMPARE,
                parameters={
                    'action': 'capture_screenshot',
                    'repeat_count': 3,
                    'delay_ms': delay_ms,
                    'compare_modality': 'vision',
                },
                expected_response={
                    'screenshots_similar': True,
                    'similarity_threshold': 0.95,
                    'no_flicker': True,
                },
                confidence_threshold=0.9,
            )
            probes.append(probe)
        
        logger.info(f"Generated {num_probes} observation comparison probes")
        return probes
    
    def generate_ui_consistency_probes(self,
                                      num_probes: int = 5) -> List[Probe]:
        """
        Generate probes that check UI consistency across modalities.
        Detects partial observation corruption.
        
        Args:
            num_probes: Number of probes
            
        Returns:
            List of Probe objects
        """
        probes = []
        
        consistency_checks = [
            'text_matches_vision',
            'button_count_consistent',
            'layout_unchanged',
            'element_positions_consistent',
            'color_scheme_consistent',
        ]
        
        for i in range(num_probes):
            check = consistency_checks[i % len(consistency_checks)]
            
            probe = Probe(
                probe_id=f"ui_consistency_{i:03d}",
                probe_type=ProbeType.UI_CONSISTENCY,
                parameters={
                    'consistency_check': check,
                    'modalities': ['vision', 'ocr'],
                    'timeout_ms': 5000,
                },
                expected_response={
                    'check_passed': True,
                    'confidence': 0.85,
                    'details': {},
                },
                confidence_threshold=0.8,
            )
            probes.append(probe)
        
        logger.info(f"Generated {num_probes} UI consistency probes")
        return probes
    
    def generate_timing_probes(self, num_probes: int = 5) -> List[Probe]:
        """
        Generate probes that check timing constraints.
        Detects replay and fast-forwarding attacks.
        
        Args:
            num_probes: Number of probes
            
        Returns:
            List of Probe objects
        """
        probes = []
        
        for i in range(num_probes):
            expected_min_time = 100 + i * 50  # ms
            expected_max_time = expected_min_time + 500
            
            probe = Probe(
                probe_id=f"timing_{i:03d}",
                probe_type=ProbeType.TIMING_CHECK,
                parameters={
                    'action': 'click_and_measure',
                    'expected_response_time_min_ms': expected_min_time,
                    'expected_response_time_max_ms': expected_max_time,
                },
                expected_response={
                    'response_time_ms': expected_min_time + 250,
                    'within_bounds': True,
                },
                confidence_threshold=0.85,
            )
            probes.append(probe)
        
        logger.info(f"Generated {num_probes} timing probes")
        return probes
    
    def generate_multimodal_sync_probes(self, num_probes: int = 5) -> List[Probe]:
        """
        Generate probes that verify audio-visual synchronization.
        Detects desynchronized multimodal attacks.
        
        Args:
            num_probes: Number of probes
            
        Returns:
            List of Probe objects
        """
        probes = []
        
        for i in range(num_probes):
            max_sync_error_ms = 50 + i * 10  # Varying thresholds
            
            probe = Probe(
                probe_id=f"multimodal_sync_{i:03d}",
                probe_type=ProbeType.MULTIMODAL_SYNC,
                parameters={
                    'trigger_action': 'button_click',
                    'modalities_to_check': ['audio', 'visual'],
                    'max_sync_error_ms': max_sync_error_ms,
                },
                expected_response={
                    'audio_visual_synced': True,
                    'sync_error_ms': 0,
                    'both_modalities_detected': True,
                },
                confidence_threshold=0.9,
            )
            probes.append(probe)
        
        logger.info(f"Generated {num_probes} multimodal sync probes")
        return probes
    
    def generate_comprehensive_probe_suite(self, 
                                          probes_per_type: int = 3) -> Dict[str, List[Probe]]:
        """
        Generate comprehensive probe suite covering all attack types.
        
        Args:
            probes_per_type: Number of probes per type
            
        Returns:
            Dict mapping probe type names to probe lists
        """
        suite = {
            'noop': self.generate_noop_probes(probes_per_type),
            'state_verify': self.generate_state_verification_probes(probes_per_type),
            'observation_compare': self.generate_observation_comparison_probes(probes_per_type),
            'ui_consistency': self.generate_ui_consistency_probes(probes_per_type),
            'timing': self.generate_timing_probes(probes_per_type),
            'multimodal_sync': self.generate_multimodal_sync_probes(probes_per_type),
        }
        
        total_probes = sum(len(probes) for probes in suite.values())
        logger.info(f"Generated comprehensive probe suite: {total_probes} total probes")
        
        return suite
    
    def save_probe_suite(self, suite: Dict[str, List[Probe]], filename: str = "probe_suite.json"):
        """
        Save probe suite to JSON file.
        
        Args:
            suite: Probe suite dict
            filename: Output filename
        """
        output_path = os.path.join(self.output_dir, filename)
        
        # Convert probes to serializable dicts
        serializable_suite = {}
        for probe_type, probes in suite.items():
            serializable_suite[probe_type] = [
                {
                    'probe_id': p.probe_id,
                    'probe_type': p.probe_type.value,
                    'parameters': p.parameters,
                    'expected_response': p.expected_response,
                    'confidence_threshold': p.confidence_threshold,
                }
                for p in probes
            ]
        
        with open(output_path, 'w') as f:
            json.dump(serializable_suite, f, indent=2)
        
        logger.info(f"Saved probe suite to {output_path}")
    
    def _generate_mock_elements(self) -> List[Dict[str, Any]]:
        """Generate mock UI elements for testing"""
        return [
            {'id': 'btn_next', 'type': 'button', 'text': 'Next', 'location': [400, 600]},
            {'id': 'inp_search', 'type': 'input', 'text': 'Search box', 'location': [200, 100]},
            {'id': 'txt_title', 'type': 'text', 'text': 'Page Title', 'location': [100, 50]},
            {'id': 'img_logo', 'type': 'image', 'text': 'Logo', 'location': [50, 50]},
            {'id': 'list_items', 'type': 'list', 'text': 'Item list', 'location': [100, 200]},
        ]


def generate_probe_suite(
    output_dir: str = "./data/probes",
    probes_per_type: int = 5,
    save_to_file: bool = True
) -> Dict[str, List[Probe]]:
    """
    Convenience function to generate comprehensive probe suite.
    
    Args:
        output_dir: Output directory
        probes_per_type: Probes per type
        save_to_file: Whether to save to JSON
        
    Returns:
        Generated probe suite
    """
    generator = ProbeGenerator(output_dir=output_dir)
    suite = generator.generate_comprehensive_probe_suite(probes_per_type)
    
    if save_to_file:
        generator.save_probe_suite(suite)
    
    return suite


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Generate probe suite
    suite = generate_probe_suite(
        output_dir="./data/probes",
        probes_per_type=3,
        save_to_file=True
    )
    
    print(f"Generated probe suite with {sum(len(p) for p in suite.values())} total probes:")
    for probe_type, probes in suite.items():
        print(f"  {probe_type}: {len(probes)} probes")
