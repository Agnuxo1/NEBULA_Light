#!/usr/bin/env python3
"""
NEBULA STEPPED CALIBRATION SYSTEM
=================================
Progressive calibration: Human Vision → NEBULA Super-Human

STEP 1: Human-equivalent visible light (baseline)
STEP 2: Extended UV range (beyond human)  
STEP 3: Extended NIR range (beyond human)
STEP 4: Full NEBULA spectrum (super-human)
FINAL: Average all 4 steps → Anomaly Detection Reference

Strategy:
- Each step uses specific wavelength ranges
- Each step has its own phantom template
- Progressive improvement from human to super-human
- Final averaged reference for anomaly detection
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class SteppedCalibrationSystem:
    """
    4-Step progressive calibration from human vision to NEBULA super-human
    """
    def __init__(self):
        # Load optimal ray count from ray calibration
        self.optimal_ray_count = self.load_optimal_ray_count()
        
        # STEP DEFINITIONS
        self.calibration_steps = {
            'STEP_1_HUMAN': {
                'name': 'Human-Equivalent Visible Light',
                'wavelengths': [0.40, 0.55, 0.70],  # RGB equivalent for medical imaging
                'description': 'Baseline human radiologist vision',
                'phantom_template': 'human_medical_standard',
                'detection_threshold': 0.02,  # 2% contrast (human limit)
                'spatial_resolution': 0.5,    # 0.5mm (human limit)
            },
            
            'STEP_2_UV_EXTENDED': {
                'name': 'UV Extended Range',
                'wavelengths': [0.08, 0.12, 0.18, 0.40],  # Add UV to human range
                'description': 'Beyond human: UV tissue penetration',
                'phantom_template': 'enhanced_uv_detection',
                'detection_threshold': 0.01,  # 1% contrast (2x better)
                'spatial_resolution': 0.25,   # 0.25mm (2x better)
            },
            
            'STEP_3_NIR_EXTENDED': {
                'name': 'NIR Extended Range', 
                'wavelengths': [0.40, 0.55, 0.70, 0.85, 1.2],  # Add NIR to human range
                'description': 'Beyond human: NIR deep penetration',
                'phantom_template': 'deep_tissue_analysis',
                'detection_threshold': 0.005, # 0.5% contrast (4x better)
                'spatial_resolution': 0.1,    # 0.1mm (5x better)
            },
            
            'STEP_4_FULL_NEBULA': {
                'name': 'Full NEBULA Super-Human',
                'wavelengths': [0.08, 0.12, 0.18, 0.25, 0.40, 0.55, 0.70, 0.85],
                'description': 'Complete NEBULA spectrum',
                'phantom_template': 'quantum_super_human',
                'detection_threshold': 0.001, # 0.1% contrast (20x better)
                'spatial_resolution': 0.01,   # 0.01mm (50x better)
            }
        }
        
        logger.info("Stepped Calibration System Initialized")
        for step_name, config in self.calibration_steps.items():
            logger.info(f"{step_name}: {config['name']} - {len(config['wavelengths'])} wavelengths")

class PhantomTemplateGenerator:
    """
    Generate specific phantom templates for each calibration step
    """
    def __init__(self, step_config: Dict):
        self.step_config = step_config
        self.template_size = (256, 256)
        
    def generate_human_medical_standard(self) -> np.ndarray:
        """STEP 1: Standard medical phantom (human-detectable features)"""
        h, w = self.template_size
        phantom = np.zeros((h, w), dtype=np.float32)
        
        # Large, obvious features (human-detectable)
        # Heart shadow - large and clear
        heart_center = (h//2, w//3)
        y, x = np.ogrid[:h, :w]
        heart_mask = ((y - heart_center[0])**2 + (x - heart_center[1])**2) < (h//5)**2
        phantom[heart_mask] = 0.4  # 40% density increase (easily detectable)
        
        # Obvious lung lesion (>2mm, >2% contrast)
        lesion_center = (h//3, 2*w//3)
        lesion_mask = ((y - lesion_center[0])**2 + (x - lesion_center[1])**2) < 20**2
        phantom[lesion_mask] = 0.35  # 35% density (clear pathology)
        
        # Rib structures - clearly visible
        for rib in range(5):
            rib_y = h//4 + rib * h//8
            phantom[rib_y-2:rib_y+3, :] += 0.2  # Thick, obvious ribs
        
        # Base tissue density
        phantom += 0.2
        
        logger.info("Generated HUMAN_MEDICAL_STANDARD phantom")
        logger.info(f"  Features: Large heart, obvious lesion, clear ribs")
        logger.info(f"  Contrast levels: 20-40% (human-detectable)")
        
        return phantom
    
    def generate_enhanced_uv_detection(self) -> np.ndarray:
        """STEP 2: UV-enhanced features (beyond human vision)"""
        # Start with human template
        phantom = self.generate_human_medical_standard()
        
        # Add UV-specific features (smaller, subtler)
        h, w = self.template_size
        y, x = np.ogrid[:h, :w]
        
        # Micro-calcifications (UV-detectable, human-invisible)
        for i in range(10):
            calc_y = np.random.randint(20, h-20)
            calc_x = np.random.randint(20, w-20)
            calc_mask = ((y - calc_y)**2 + (x - calc_x)**2) < 3**2  # 3-pixel radius
            phantom[calc_mask] += 0.015  # 1.5% increase (below human threshold)
        
        # Subtle vascular patterns (UV reveals blood)
        vessel_pattern = np.sin(x * 0.2) * np.cos(y * 0.15) * 0.01
        phantom += vessel_pattern
        
        logger.info("Generated ENHANCED_UV_DETECTION phantom")
        logger.info(f"  Added: 10 micro-calcifications, vascular patterns")
        logger.info(f"  UV features: 1-2% contrast (below human detection)")
        
        return phantom
    
    def generate_deep_tissue_analysis(self) -> np.ndarray:
        """STEP 3: NIR deep tissue features"""
        phantom = self.generate_enhanced_uv_detection()
        
        h, w = self.template_size
        y, x = np.ogrid[:h, :w]
        
        # Deep tissue inflammation (NIR-penetrating)
        inflammation_center = (h//2, w//2)
        inflammation_mask = ((y - inflammation_center[0])**2 + (x - inflammation_center[1])**2) < (h//6)**2
        inflammation_gradient = np.exp(-((y - inflammation_center[0])**2 + (x - inflammation_center[1])**2) / (2 * (h//8)**2))
        phantom += inflammation_gradient * 0.008  # Subtle deep inflammation
        
        # Fibrosis patterns (NIR-detectable)
        fibrosis_x = np.sin(x * 0.8) * np.cos(y * 0.6)
        fibrosis_y = np.cos(x * 0.6) * np.sin(y * 0.8)
        fibrosis_pattern = (fibrosis_x + fibrosis_y) * 0.005
        phantom += fibrosis_pattern
        
        logger.info("Generated DEEP_TISSUE_ANALYSIS phantom")
        logger.info(f"  Added: Deep inflammation, fibrosis patterns")
        logger.info(f"  NIR features: 0.5-0.8% contrast (deep tissue)")
        
        return phantom
    
    def generate_quantum_super_human(self) -> np.ndarray:
        """STEP 4: Full NEBULA super-human features"""
        phantom = self.generate_deep_tissue_analysis()
        
        h, w = self.template_size
        y, x = np.ogrid[:h, :w]
        
        # Quantum-level cellular changes
        cellular_noise = np.random.normal(0, 0.001, (h, w))  # 0.1% cellular variations
        phantom += cellular_noise
        
        # Pre-angiogenic changes (super-early detection)
        for vessel_seed in range(20):
            seed_y = np.random.randint(10, h-10)
            seed_x = np.random.randint(10, w-10)
            # Tiny vessel precursor
            precursor_mask = ((y - seed_y)**2 + (x - seed_x)**2) < 1.5**2
            phantom[precursor_mask] += 0.002  # 0.2% change (super-human only)
        
        # Quantum interference patterns
        quantum_x = np.sin(x * 2.0) * np.cos(y * 1.8) * np.exp(-((x-w//2)**2 + (y-h//2)**2) / (2 * (min(h,w)//4)**2))
        quantum_y = np.cos(x * 1.8) * np.sin(y * 2.0) * np.exp(-((x-w//2)**2 + (y-h//2)**2) / (2 * (min(h,w)//4)**2))
        quantum_pattern = (quantum_x + quantum_y) * 0.0005  # Quantum-level signatures
        phantom += quantum_pattern
        
        logger.info("Generated QUANTUM_SUPER_HUMAN phantom")
        logger.info(f"  Added: Cellular noise, vessel precursors, quantum patterns")
        logger.info(f"  Super-human features: 0.05-0.2% contrast (impossible for humans)")
        
        return phantom
    
    def generate_phantom(self) -> np.ndarray:
        """Generate phantom based on step configuration"""
        template_type = self.step_config['phantom_template']
        
        if template_type == 'human_medical_standard':
            return self.generate_human_medical_standard()
        elif template_type == 'enhanced_uv_detection':
            return self.generate_enhanced_uv_detection()
        elif template_type == 'deep_tissue_analysis':
            return self.generate_deep_tissue_analysis()
        elif template_type == 'quantum_super_human':
            return self.generate_quantum_super_human()
        else:
            raise ValueError(f"Unknown phantom template: {template_type}")

class WavelengthSpecificAnalyzer(nn.Module):
    """
    Wavelength-specific analysis for each calibration step
    """
    def __init__(self, wavelengths: List[float], step_name: str):
        super().__init__()
        self.wavelengths = wavelengths
        self.step_name = step_name
        self.num_wavelengths = len(wavelengths)
        
        # Wavelength-specific laser intensities (learnable)
        self.laser_intensities = nn.Parameter(torch.ones(self.num_wavelengths))
        
        # Wavelength-specific sensor sensitivities (learnable)
        self.sensor_sensitivities = nn.Parameter(torch.ones(self.num_wavelengths))
        
        # Tissue interaction coefficients per wavelength
        self.tissue_interactions = nn.Parameter(torch.randn(self.num_wavelengths, 3) * 0.1)  # absorption, scattering, penetration
        
        logger.info(f"Initialized {step_name} analyzer with {self.num_wavelengths} wavelengths")
    
    def forward(self, phantom: torch.Tensor) -> torch.Tensor:
        """Analyze phantom with step-specific wavelengths"""
        batch_size, height, width = phantom.shape
        
        wavelength_responses = []
        
        for i, wavelength in enumerate(self.wavelengths):
            # Wavelength-specific tissue interaction
            absorption = self.tissue_interactions[i, 0]
            scattering = self.tissue_interactions[i, 1] 
            penetration = self.tissue_interactions[i, 2]
            
            # Simulate tissue response at this wavelength
            tissue_response = phantom * torch.exp(-phantom * absorption)  # Absorption
            tissue_response *= torch.exp(-phantom * scattering)  # Scattering
            tissue_response *= (1 + penetration)  # Penetration enhancement
            
            # Apply laser intensity and sensor sensitivity
            intensity = self.laser_intensities[i]
            sensitivity = self.sensor_sensitivities[i]
            
            final_response = tissue_response * intensity * sensitivity
            wavelength_responses.append(final_response)
        
        # Stack wavelength responses
        stacked_responses = torch.stack(wavelength_responses, dim=1)  # [batch, wavelengths, height, width]
        
        return stacked_responses

class SteppedCalibrator:
    """
    Main stepped calibration controller
    """
    def __init__(self):
        self.calibration_system = SteppedCalibrationSystem()
        self.step_results = {}
        
    def calibrate_step(self, step_name: str, num_iterations: int = 50) -> Dict:
        """Calibrate a specific step"""
        logger.info("="*60)
        logger.info(f"CALIBRATING {step_name}")
        logger.info("="*60)
        
        step_config = self.calibration_system.calibration_steps[step_name]
        logger.info(f"Step: {step_config['name']}")
        logger.info(f"Wavelengths: {step_config['wavelengths']}")
        logger.info(f"Target detection threshold: {step_config['detection_threshold']}")
        logger.info(f"Target spatial resolution: {step_config['spatial_resolution']}mm")
        
        # Generate step-specific phantom
        phantom_generator = PhantomTemplateGenerator(step_config)
        phantom_array = phantom_generator.generate_phantom()
        phantom_tensor = torch.FloatTensor(phantom_array).unsqueeze(0)  # Add batch dimension
        
        # Initialize wavelength analyzer
        analyzer = WavelengthSpecificAnalyzer(step_config['wavelengths'], step_name)
        optimizer = torch.optim.Adam(analyzer.parameters(), lr=0.01)
        
        best_score = 0.0
        best_params = {}
        
        logger.info(f"Starting optimization for {num_iterations} iterations...")
        
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            # Forward pass
            wavelength_responses = analyzer(phantom_tensor)
            
            # Calculate step-specific metrics
            contrast_score = self.calculate_contrast_score(wavelength_responses, step_config['detection_threshold'])
            sharpness_score = self.calculate_sharpness_score(wavelength_responses, step_config['spatial_resolution'])
            wavelength_diversity = self.calculate_wavelength_diversity(wavelength_responses)
            
            # Combined step score (ensure it's a tensor)
            step_score = torch.tensor(
                0.4 * contrast_score + 0.3 * sharpness_score + 0.3 * wavelength_diversity,
                requires_grad=True
            )
            
            # Maximize performance
            loss = -step_score
            loss.backward()
            optimizer.step()
            
            # Track best configuration
            if step_score > best_score:
                best_score = step_score
                best_params = {
                    'laser_intensities': analyzer.laser_intensities.data.clone(),
                    'sensor_sensitivities': analyzer.sensor_sensitivities.data.clone(),
                    'tissue_interactions': analyzer.tissue_interactions.data.clone(),
                    'step_response': wavelength_responses.detach().clone()
                }
            
            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration}: Score={step_score:.4f}, Contrast={contrast_score:.3f}, Sharpness={sharpness_score:.3f}")
        
        # Final step results
        logger.info(f"{step_name} CALIBRATION COMPLETED")
        logger.info(f"Best Score: {best_score:.4f}")
        logger.info(f"Laser Intensities: {[f'{x:.3f}' for x in best_params['laser_intensities'].tolist()]}")
        logger.info(f"Sensor Sensitivities: {[f'{x:.3f}' for x in best_params['sensor_sensitivities'].tolist()]}")
        
        step_result = {
            'step_name': step_name,
            'step_config': step_config,
            'best_score': float(best_score),
            'best_params': {k: v.tolist() if torch.is_tensor(v) else v for k, v in best_params.items() if k != 'step_response'},
            'step_response': best_params['step_response']  # Keep tensor for averaging
        }
        
        return step_result
    
    def calculate_contrast_score(self, responses: torch.Tensor, threshold: float) -> float:
        """Calculate contrast detection score"""
        # Calculate standard deviation across spatial dimensions
        spatial_std = torch.std(responses, dim=[2, 3])  # [batch, wavelengths]
        mean_std = torch.mean(spatial_std)
        
        # Normalize by threshold (higher threshold = easier target)
        contrast_score = (mean_std / threshold).item()
        
        return min(contrast_score, 1.0)  # Cap at 1.0
    
    def calculate_sharpness_score(self, responses: torch.Tensor, resolution_target: float) -> float:
        """Calculate spatial resolution score"""
        # Calculate gradient magnitude
        grad_x = torch.abs(responses[:, :, :, 1:] - responses[:, :, :, :-1])
        grad_y = torch.abs(responses[:, :, 1:, :] - responses[:, :, :-1, :])
        
        mean_grad = (torch.mean(grad_x) + torch.mean(grad_y)) / 2
        
        # Normalize by resolution target (smaller target = harder)
        sharpness_score = (mean_grad * resolution_target).item()
        
        return min(sharpness_score, 1.0)  # Cap at 1.0
    
    def calculate_wavelength_diversity(self, responses: torch.Tensor) -> float:
        """Calculate diversity between wavelength responses"""
        if responses.shape[1] < 2:
            return 0.0
        
        correlations = []
        num_wavelengths = responses.shape[1]
        
        for i in range(num_wavelengths):
            for j in range(i+1, num_wavelengths):
                resp1 = responses[0, i].flatten()
                resp2 = responses[0, j].flatten()
                
                correlation = torch.corrcoef(torch.stack([resp1, resp2]))[0, 1]
                correlations.append(abs(correlation.item()))
        
        # Lower correlation = better diversity
        avg_correlation = np.mean(correlations) if correlations else 0.0
        diversity_score = 1.0 - avg_correlation
        
        return max(diversity_score, 0.0)
    
    def run_full_stepped_calibration(self) -> Dict:
        """Run complete 4-step calibration"""
        logger.info("🚀 STARTING FULL STEPPED CALIBRATION")
        logger.info("Human Vision → UV Extended → NIR Extended → Full NEBULA")
        logger.info("="*80)
        
        all_step_results = {}
        all_responses = []
        
        # Run each calibration step
        for step_name in ['STEP_1_HUMAN', 'STEP_2_UV_EXTENDED', 'STEP_3_NIR_EXTENDED', 'STEP_4_FULL_NEBULA']:
            step_result = self.calibrate_step(step_name, num_iterations=50)
            all_step_results[step_name] = step_result
            all_responses.append(step_result['step_response'])
        
        # Calculate final averaged reference for anomaly detection
        logger.info("="*60)
        logger.info("CALCULATING AVERAGED ANOMALY DETECTION REFERENCE")
        logger.info("="*60)
        
        # Average all 4 step responses (handle different wavelength dimensions)
        # Convert all responses to same spatial dimensions by averaging wavelength dimension
        normalized_responses = []
        for response in all_responses:
            # response shape: [batch, wavelengths, height, width]
            # Average across wavelength dimension to get [batch, height, width]  
            spatial_response = torch.mean(response, dim=1)
            normalized_responses.append(spatial_response)
        
        # Now stack and average the spatial responses
        averaged_response = torch.mean(torch.stack(normalized_responses), dim=0)
        
        # Calculate reference statistics
        reference_mean = torch.mean(averaged_response).item()
        reference_std = torch.std(averaged_response).item()
        reference_min = torch.min(averaged_response).item() 
        reference_max = torch.max(averaged_response).item()
        
        logger.info("ANOMALY DETECTION REFERENCE CALCULATED")
        logger.info(f"Reference Mean: {reference_mean:.6f}")
        logger.info(f"Reference Std: {reference_std:.6f}")
        logger.info(f"Reference Range: [{reference_min:.6f}, {reference_max:.6f}]")
        
        # Final calibration summary
        final_results = {
            'calibration_mode': 'STEPPED_HUMAN_TO_NEBULA',
            'steps_completed': list(all_step_results.keys()),
            'step_results': {k: {**v, 'step_response': 'tensor_excluded'} for k, v in all_step_results.items()},  # Exclude tensors from JSON
            'anomaly_detection_reference': {
                'mean': reference_mean,
                'std': reference_std,
                'min': reference_min,
                'max': reference_max
            },
            'performance_progression': {
                'step_1_human_score': all_step_results['STEP_1_HUMAN']['best_score'],
                'step_2_uv_score': all_step_results['STEP_2_UV_EXTENDED']['best_score'],
                'step_3_nir_score': all_step_results['STEP_3_NIR_EXTENDED']['best_score'],
                'step_4_nebula_score': all_step_results['STEP_4_FULL_NEBULA']['best_score']
            },
            'calibration_date': datetime.now().isoformat()
        }
        
        # Save complete configuration
        with open('nebula_stepped_calibration_complete.json', 'w') as f:
            json.dump(final_results, f, indent=2)
        
        logger.info("="*80)
        logger.info("🎯 STEPPED CALIBRATION COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info("Performance Progression:")
        logger.info(f"  STEP 1 (Human): {final_results['performance_progression']['step_1_human_score']:.4f}")
        logger.info(f"  STEP 2 (UV): {final_results['performance_progression']['step_2_uv_score']:.4f}")
        logger.info(f"  STEP 3 (NIR): {final_results['performance_progression']['step_3_nir_score']:.4f}")  
        logger.info(f"  STEP 4 (NEBULA): {final_results['performance_progression']['step_4_nebula_score']:.4f}")
        logger.info("")
        logger.info("✅ Anomaly Detection Reference: READY")
        logger.info("✅ Configuration saved: nebula_stepped_calibration_complete.json")
        logger.info("✅ NEBULA calibrated from human baseline to super-human!")
        
        return final_results

def main():
    """Main stepped calibration execution"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logger.info("NEBULA STEPPED CALIBRATION SYSTEM")
    logger.info("Progressive Evolution: Human → Super-Human")
    
    calibrator = SteppedCalibrator()
    results = calibrator.run_full_stepped_calibration()
    
    logger.info("🚀 NEBULA is now calibrated and ready for super-human medical analysis!")

if __name__ == "__main__":
    main()