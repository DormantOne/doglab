#!/usr/bin/env python3
"""
================================================================================
🦿 DOG LAB v7 — "IRON WOLF 3D" (ENHANCED)
   
   IMPROVEMENTS OVER v6:
   ----------------------
   1. CPG (Central Pattern Generator) provides rhythmic gait baseline
   2. Network learns RESIDUALS on CPG, not raw actions (much easier!)
   3. Coupled physics - leg movements create proper body reactions
   4. Gait coordination rewards (diagonal sync, symmetry, smoothness)
   5. Soft joint limits with damping (no jerky bouncing)
   6. Extended training time and better hyperparameters
   7. Height stability and lateral drift penalties
   
   Physics: 3D Quadruped with Mirror Floor Contact Model
   Brain:   Multi-timescale Liquid Reservoir + Actor-Critic (A2C style)
   Search:  Natural-Gradient-ish Evolution (adaptive per-gene sigma) + diversity
   Metric:  PRIMARY = distance traveled (mean_distance)
   
   VISUALIZATION: Three.js with mouse-controlled orbit camera
================================================================================

Run:
  pip install flask torch numpy
  python dog_lab_v7_3d.py
  
Controls:
  - Left mouse drag: Rotate camera
  - Right mouse drag: Pan camera
  - Scroll wheel: Zoom in/out
  - R key: Recenter camera on dog
================================================================================
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import threading
import time
import multiprocessing as mp
import webbrowser
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from concurrent.futures import ProcessPoolExecutor, as_completed
from flask import Flask, jsonify, Response, request

# =============================================================================
# CONFIG - UPDATED HYPERPARAMETERS
# =============================================================================

DEVICE = torch.device("cpu")
logging.getLogger("werkzeug").setLevel(logging.ERROR)

CORE_COUNT = max(1, min(6, os.cpu_count() or 1))
POP_MULT = 4
POP_SIZE = CORE_COUNT * POP_MULT

HYPER_D = 9
REWARD_D = 5
GENOME_DIM = HYPER_D + REWARD_D

GENE_NAMES = ["SIZE", "DENS", "LK_F", "LK_M", "LK_S",
              "PL_F", "PL_S", "RAD", "LR",
              "W_ALV", "W_DST", "W_UP", "W_TAU", "W_CNT"]

MIN_RESERVOIR_SIZE = 400
MAX_RESERVOIR_SIZE = 1400
MIN_DENSITY = 0.08
MAX_SPECTRAL_RADIUS = 1.5
MIN_ALIVE_WEIGHT = 0.15

# === TRAINING HYPERPARAMS (BALANCED) ===
# Increased from v6 but not so much it takes forever
TRAIN_EPISODES_INITIAL = 30   # Was 25 in v6, 60 was too slow
TRAIN_EPISODES_VARSITY = 60   # Was 90 in v6, 180 was way too slow
VARSITY_DISTANCE_THRESHOLD = 0.35  # Threshold to get extra training
SCOUT_EPISODES = 2   # Quick scout
EVAL_EPISODES = 4    # Reasonable eval

DT_PHYS = 0.002
CONTROL_DT = 0.02
SUBSTEPS = int(round(CONTROL_DT / DT_PHYS))
SUBSTEPS = max(1, SUBSTEPS)
MAX_STEPS = 400  # Shorter episodes for faster iteration

TAU_MAX = 20.0  # Was 25.0 - less extreme torques

TORSO_DOF = 6
LEGS_DOF = 12
TOTAL_DOF = TORSO_DOF + LEGS_DOF

# === OBSERVATION DIMENSION (INCREASED FOR CPG SIGNALS) ===
# Original: 49
# Added: 8 CPG phase signals (swing/stance for each leg)
# Added: 4 CPG target differences (how far from CPG baseline)
OBS_DIM = 61  # Was 49

ACTION_DIM = 12
INPUT_DIM = OBS_DIM + ACTION_DIM

GAMMA = 0.99
GRAD_CLIP = 1.0
VALUE_COEF = 0.25    # Was 0.50 - less value function pressure
ENTROPY_COEF = 0.01  # Was 0.003 - more exploration early on

IP_LEARNING_RATE = 0.001
IP_TARGET_MEAN = 0.0
IP_TARGET_VAR = 0.2

INIT_SIGMA = 0.06
SIGMA_MIN = 0.004
SIGMA_MAX = 0.20

SENSITIVITY_SMOOTHING = 0.85
SENSITIVITY_FLOOR = 0.01
SENSITIVITY_THRESHOLD = 0.5

SIGMA_GROW_RATE = 1.08
SIGMA_SHRINK_RATE = 0.92

N_ELITES = 5
CROSSOVER_PROB = 0.3
BLX_ALPHA = 0.3

DIVERSITY_INJECT_EVERY = 50
DIVERSITY_INJECT_COUNT = 2
STAGNATION_THRESHOLD = 30
STAGNATION_SIGMA_BOOST = 1.5

VIZ_MAX_NEURONS = 200
VIZ_MAX_LINKS = 400

SAVE_FILE = "dog_lab_3d_best_v7.json"
WEIGHTS_FILE = "dog_lab_3d_best_v7.pt"
SIGMA_FILE = "dog_lab_3d_sigma_v7.json"
BEST_METRIC_TAG = "mean_distance_3d_v7"

DEFAULT_KP = 50.0
DEFAULT_KD = 3.0

# === CPG PARAMETERS ===
CPG_FREQUENCY = 2.0      # Hz - gait cycle frequency
CPG_DUTY_CYCLE = 0.60    # Fraction of cycle spent in stance
RESIDUAL_SCALE = 0.35    # How much network can adjust CPG targets (radians)

# === COORDINATION REWARD WEIGHTS (fixed, not evolved) ===
COORD_WEIGHT_SYMMETRY = 0.15
COORD_WEIGHT_SMOOTH = 0.12
COORD_WEIGHT_STABLE = 0.25
COORD_WEIGHT_LATERAL = 0.15
COORD_WEIGHT_DIAG_GAIT = 0.20

# =============================================================================
# SHARED STATE
# =============================================================================

SYSTEM_STATE: Dict[str, Any] = {
    "status": "INITIALIZING…",
    "generation": 0,
    "best_distance": -1e9,
    "best_steps": 0.0,
    "best_genome": None,
    "best_weights": None,
    "mode": "TRAINING",
    "logs": [],
    "hyperparams": {},
    "current_id": "Waiting…",
    "manual_demo_request": False,
    "demo_stop": False,
    "demo_resets": 0,
    "pop_vectors": [],
    "history_vectors": [],
    "sim_view": {},
    "brain_view": {},
    "viz_mode": "hyperspace",
    "gene_sigma": None,
    "gene_sensitivity": None,
    "stagnation_count": 0,
    "kp": DEFAULT_KP,
    "kd": DEFAULT_KD,
}

_STATE_LOCK = threading.RLock()


def add_log(msg: str) -> None:
    with _STATE_LOCK:
        SYSTEM_STATE["logs"].insert(0, msg)
        if len(SYSTEM_STATE["logs"]) > 25:
            SYSTEM_STATE["logs"].pop()
    print(f"[DOG_LAB_3D_v7] {msg}")


# =============================================================================
# 3D MATH UTILITIES
# =============================================================================

def rotation_matrix_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Create rotation matrix from Euler angles (roll, pitch, yaw).
    Uses XYZ convention: roll around X, pitch around Y, yaw around Z.
    """
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    
    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr]
    ], dtype=np.float64)
    
    return R


def rotate_point(R: np.ndarray, point: np.ndarray) -> np.ndarray:
    return R @ point


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


# =============================================================================
# CENTRAL PATTERN GENERATOR (CPG) - NEW!
# =============================================================================

class CPGPrior:
    """
    Central Pattern Generator that provides rhythmic baseline signals for locomotion.
    
    The key insight: instead of learning locomotion from scratch (HARD),
    the neural network learns small adjustments to this baseline (EASY).
    
    This implements a trotting gait where diagonal leg pairs move together:
    - FL (front-left) and BR (back-right) are in phase
    - FR (front-right) and BL (back-left) are in phase
    - The two diagonal pairs are 180° out of phase
    """
    
    def __init__(self, freq: float = CPG_FREQUENCY, duty_cycle: float = CPG_DUTY_CYCLE):
        self.freq = freq  # Gait cycle frequency in Hz
        self.duty_cycle = duty_cycle  # Fraction of cycle in stance phase
        
        # Phase offsets for trotting gait (diagonal pairs synchronized)
        self.phase_offsets = {
            'FL': 0.0,    # Front-left: reference phase
            'FR': 0.5,    # Front-right: opposite phase (trot)
            'BL': 0.5,    # Back-left: same as FR (diagonal pair)
            'BR': 0.0,    # Back-right: same as FL (diagonal pair)
        }
        
        # Leg indices for array access
        self.leg_indices = {'FL': 0, 'FR': 1, 'BL': 2, 'BR': 3}
        
        self.t = 0.0
        
        # Joint angle ranges for the gait
        # These define the swing/stance positions for each joint
        self.swing_angles = {
            'abd': 0.0,    # Abduction: neutral during swing
            'hip': 0.5,    # Hip: forward during swing
            'knee': -1.2,  # Knee: bent during swing (negative = flexed)
        }
        self.stance_angles = {
            'abd': 0.0,    # Abduction: neutral during stance
            'hip': -0.1,   # Hip: backward during stance (pushoff)
            'knee': -0.5,  # Knee: more extended during stance
        }
        self.mid_angles = {
            'abd': 0.0,
            'hip': 0.25,
            'knee': -0.7,
        }
    
    def reset(self) -> None:
        """Reset CPG time to zero."""
        self.t = 0.0
    
    def step(self, dt: float) -> None:
        """Advance CPG time."""
        self.t += dt
    
    def get_phase(self, leg: str) -> float:
        """
        Get current phase [0, 1) for a specific leg.
        Phase 0 = start of stance, Phase duty_cycle = start of swing.
        """
        offset = self.phase_offsets[leg]
        phase = (self.t * self.freq + offset) % 1.0
        return phase
    
    def get_signals(self) -> Dict[str, Tuple[float, float]]:
        """
        Get CPG signals for each leg.
        
        Returns:
            Dict mapping leg name to (swing_progress, stance_progress)
            where each progress value is 0-1 during that phase, 0 otherwise.
        """
        signals = {}
        for leg in ['FL', 'FR', 'BL', 'BR']:
            phase = self.get_phase(leg)
            
            if phase < self.duty_cycle:
                # Stance phase: foot on ground, pushing back
                stance_progress = phase / self.duty_cycle
                swing_progress = 0.0
            else:
                # Swing phase: foot in air, moving forward
                swing_progress = (phase - self.duty_cycle) / (1.0 - self.duty_cycle)
                stance_progress = 0.0
            
            signals[leg] = (swing_progress, stance_progress)
        
        return signals
    
    def get_target_angles(self) -> np.ndarray:
        """
        Compute target joint angles for all legs based on current CPG phase.
        
        The network will output RESIDUALS to add to these targets,
        rather than computing raw angles from scratch.
        
        Returns:
            Array of shape (12,) with target angles for all joints:
            [FL_abd, FL_hip, FL_knee, FR_abd, FR_hip, FR_knee, 
             BL_abd, BL_hip, BL_knee, BR_abd, BR_hip, BR_knee]
        """
        targets = np.zeros(12, dtype=np.float64)
        signals = self.get_signals()
        
        for leg in ['FL', 'FR', 'BL', 'BR']:
            idx = self.leg_indices[leg]
            base = idx * 3
            
            swing_progress, stance_progress = signals[leg]
            
            if swing_progress > 0:
                # === SWING PHASE ===
                # Smooth trajectory from stance-end to swing-peak to stance-start
                # Use sinusoidal profile for smooth motion
                
                # First half of swing: lift and move forward
                if swing_progress < 0.5:
                    t = swing_progress * 2  # 0 to 1 in first half
                    # Interpolate from stance-end to swing-peak
                    hip_angle = self.stance_angles['hip'] + \
                               (self.swing_angles['hip'] - self.stance_angles['hip']) * \
                               (1 - math.cos(t * math.pi)) / 2
                    knee_angle = self.stance_angles['knee'] + \
                                (self.swing_angles['knee'] - self.stance_angles['knee']) * \
                                (1 - math.cos(t * math.pi)) / 2
                else:
                    # Second half of swing: move down to prepare for contact
                    t = (swing_progress - 0.5) * 2  # 0 to 1 in second half
                    # Interpolate from swing-peak to mid-stance
                    hip_angle = self.swing_angles['hip'] + \
                               (self.mid_angles['hip'] - self.swing_angles['hip']) * \
                               (1 - math.cos(t * math.pi)) / 2
                    knee_angle = self.swing_angles['knee'] + \
                                (self.mid_angles['knee'] - self.swing_angles['knee']) * \
                                (1 - math.cos(t * math.pi)) / 2
                
                abd_angle = self.swing_angles['abd']
                
            else:
                # === STANCE PHASE ===
                # Foot on ground, leg moves backward relative to body
                t = stance_progress
                
                # Linear interpolation during stance (pushing motion)
                hip_angle = self.mid_angles['hip'] + \
                           (self.stance_angles['hip'] - self.mid_angles['hip']) * t
                knee_angle = self.mid_angles['knee'] + \
                            (self.stance_angles['knee'] - self.mid_angles['knee']) * t
                abd_angle = self.stance_angles['abd']
            
            targets[base + 0] = abd_angle
            targets[base + 1] = hip_angle
            targets[base + 2] = knee_angle
        
        return targets
    
    def get_phase_encoding(self) -> np.ndarray:
        """
        Get phase encoding for observation space.
        
        Returns sin/cos encoding of each leg's phase for smooth interpolation.
        """
        encoding = []
        for leg in ['FL', 'FR', 'BL', 'BR']:
            phase = self.get_phase(leg)
            # Sin/cos encoding prevents discontinuity at phase wrap
            encoding.append(math.sin(2 * math.pi * phase))
            encoding.append(math.cos(2 * math.pi * phase))
        return np.array(encoding, dtype=np.float64)


# =============================================================================
# 3D PHYSICS MODEL - ENHANCED WITH COUPLED DYNAMICS
# =============================================================================

@dataclass
class DogConfig3D:
    """Physical parameters for the quadruped robot."""
    torso_length: float = 0.50
    torso_width: float = 0.20
    torso_height: float = 0.12
    thigh_length: float = 0.22
    calf_length: float = 0.22
    hip_offset_x: float = 0.20
    hip_offset_z: float = 0.12
    hip_offset_y: float = -0.04
    torso_mass: float = 8.0
    thigh_mass: float = 0.8
    calf_mass: float = 0.4
    torso_Ixx: float = 0.05
    torso_Iyy: float = 0.08
    torso_Izz: float = 0.06
    floor_k: float = 15000.0
    floor_b: float = 300.0
    mu: float = 0.9
    g: float = 9.81
    dt: float = DT_PHYS


class MirrorFloor3D:
    """
    Ground contact model using "shadow" anchors.
    
    When a foot penetrates the ground, a shadow point is created.
    The shadow acts like a spring anchor, providing:
    - Vertical spring-damper force (support)
    - Horizontal spring force (friction via anchor displacement)
    
    If horizontal force exceeds friction limit, the shadow slides.
    """
    
    def __init__(self, cfg: DogConfig3D):
        self.cfg = cfg
        self.shadows: Dict[str, np.ndarray] = {}
    
    def compute_force(self, point_name: str, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """
        Compute ground reaction force for a contact point.
        
        Args:
            point_name: Identifier for this contact point
            pos: 3D position [x, y, z] where y is up
            vel: 3D velocity
            
        Returns:
            Force vector [fx, fy, fz]
        """
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        vx, vy, vz = float(vel[0]), float(vel[1]), float(vel[2])
        
        # Above ground - no contact
        if y >= 0:
            if point_name in self.shadows:
                del self.shadows[point_name]
            return np.array([0.0, 0.0, 0.0], dtype=np.float64)
        
        # Create shadow if new contact
        if point_name not in self.shadows:
            self.shadows[point_name] = np.array([x, z], dtype=np.float64)
        
        shadow_x, shadow_z = self.shadows[point_name]
        
        # Vertical force: spring-damper
        penetration = -y
        fy = max(0.0, self.cfg.floor_k * penetration - self.cfg.floor_b * vy)
        
        # Friction limit (Coulomb model)
        f_max = self.cfg.mu * fy
        
        # Horizontal forces: spring to shadow anchor
        dx = shadow_x - x
        fx = self.cfg.floor_k * dx
        
        dz = shadow_z - z
        fz = self.cfg.floor_k * dz
        
        # Check if friction limit exceeded
        f_horiz = math.sqrt(fx * fx + fz * fz)
        
        if f_horiz > f_max and f_horiz > 1e-9:
            # Sliding - scale force to friction limit and update shadow
            scale = f_max / f_horiz
            fx *= scale
            fz *= scale
            # Shadow slides to maintain friction limit
            self.shadows[point_name] = np.array([
                x + fx / self.cfg.floor_k,
                z + fz / self.cfg.floor_k
            ], dtype=np.float64)
        
        # Add velocity damping for stability
        fx -= 0.1 * self.cfg.floor_b * vx
        fz -= 0.1 * self.cfg.floor_b * vz
        
        return np.array([fx, fy, fz], dtype=np.float64)


# Leg indices
LEG_FL = 0
LEG_FR = 1
LEG_BL = 2
LEG_BR = 3
LEG_NAMES = ["FL", "FR", "BL", "BR"]

# Joint indices within each leg
JOINT_ABD = 0   # Abduction (sideways hip rotation)
JOINT_HIP = 1   # Hip flexion (forward/backward)
JOINT_KNEE = 2  # Knee flexion


class DogModel3D:
    """
    3D quadruped physics simulation with coupled dynamics.
    
    Key improvement over v6: leg movements now properly affect torso dynamics
    through reaction torques and coupled inertia.
    """
    
    def __init__(self, cfg: DogConfig3D):
        self.cfg = cfg
        self.floor = MirrorFloor3D(cfg)
        self.kp = DEFAULT_KP
        self.kd = DEFAULT_KD
        self.reset()
    
    def reset(self) -> None:
        """Reset robot to standing pose."""
        self.q = np.zeros(TOTAL_DOF, dtype=np.float64)
        self.qd = np.zeros(TOTAL_DOF, dtype=np.float64)
        
        # Initial torso position - standing height
        standing_height = self.cfg.thigh_length + self.cfg.calf_length - 0.05
        self.q[0] = 0.0  # x
        self.q[1] = standing_height  # y (up)
        self.q[2] = 0.0  # z
        self.q[3] = 0.0  # roll
        self.q[4] = 0.0  # pitch
        self.q[5] = 0.0  # yaw
        
        # Initial leg pose - slightly bent
        default_leg = [0.0, 0.25, -0.7]  # abd, hip, knee
        
        for leg in range(4):
            base_idx = TORSO_DOF + leg * 3
            self.q[base_idx + JOINT_ABD] = default_leg[0]
            self.q[base_idx + JOINT_HIP] = default_leg[1]
            self.q[base_idx + JOINT_KNEE] = default_leg[2]
        
        self.t = 0.0
        self.floor.shadows = {}
        
        # Track previous joint accelerations for reaction torque calculation
        self.prev_qdd_joints = np.zeros(LEGS_DOF, dtype=np.float64)
    
    def get_hip_position(self, leg_idx: int) -> np.ndarray:
        """Get world-space position of a hip joint."""
        cfg = self.cfg
        
        # Hip offset in body frame
        if leg_idx == LEG_FL:
            offset = np.array([cfg.hip_offset_x, cfg.hip_offset_y, cfg.hip_offset_z])
        elif leg_idx == LEG_FR:
            offset = np.array([cfg.hip_offset_x, cfg.hip_offset_y, -cfg.hip_offset_z])
        elif leg_idx == LEG_BL:
            offset = np.array([-cfg.hip_offset_x, cfg.hip_offset_y, cfg.hip_offset_z])
        else:  # BR
            offset = np.array([-cfg.hip_offset_x, cfg.hip_offset_y, -cfg.hip_offset_z])
        
        # Transform to world frame
        torso_pos = self.q[0:3]
        roll, pitch, yaw = self.q[3], self.q[4], self.q[5]
        R = rotation_matrix_from_euler(roll, pitch, yaw)
        
        hip_world = torso_pos + R @ offset
        return hip_world
    
    def get_leg_kinematics(self, leg_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute forward kinematics for a leg.
        
        Returns:
            Tuple of (hip_pos, knee_pos, toe_pos) in world coordinates.
        """
        cfg = self.cfg
        
        # Get joint angles
        base_idx = TORSO_DOF + leg_idx * 3
        q_abd = self.q[base_idx + JOINT_ABD]
        q_hip = self.q[base_idx + JOINT_HIP]
        q_knee = self.q[base_idx + JOINT_KNEE]
        
        # Torso rotation matrix
        roll, pitch, yaw = self.q[3], self.q[4], self.q[5]
        R_torso = rotation_matrix_from_euler(roll, pitch, yaw)
        
        hip_pos = self.get_hip_position(leg_idx)
        
        # Abduction axis depends on which side
        is_left = (leg_idx == LEG_FL or leg_idx == LEG_BL)
        abd_sign = 1.0 if is_left else -1.0
        
        # Abduction rotation (around x-axis in body frame)
        ca, sa = math.cos(q_abd * abd_sign), math.sin(q_abd * abd_sign)
        R_abd = np.array([
            [1, 0, 0],
            [0, ca, -sa],
            [0, sa, ca]
        ], dtype=np.float64)
        
        # Hip rotation (around z-axis, flexion/extension)
        ch, sh = math.cos(q_hip), math.sin(q_hip)
        R_hip = np.array([
            [ch, -sh, 0],
            [sh, ch, 0],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Combined rotation for thigh
        R_thigh = R_torso @ R_abd @ R_hip
        
        # Thigh vector (points downward in leg frame)
        thigh_vec_local = np.array([0.0, -cfg.thigh_length, 0.0])
        thigh_vec_world = R_thigh @ thigh_vec_local
        
        knee_pos = hip_pos + thigh_vec_world
        
        # Knee rotation
        ck, sk = math.cos(q_knee), math.sin(q_knee)
        R_knee = np.array([
            [ck, -sk, 0],
            [sk, ck, 0],
            [0, 0, 1]
        ], dtype=np.float64)
        
        R_calf = R_thigh @ R_knee
        
        # Calf vector
        calf_vec_local = np.array([0.0, -cfg.calf_length, 0.0])
        calf_vec_world = R_calf @ calf_vec_local
        
        toe_pos = knee_pos + calf_vec_world
        
        return hip_pos, knee_pos, toe_pos
    
    def compute_leg_jacobian(self, leg_idx: int) -> np.ndarray:
        """
        Compute Jacobian mapping joint velocities to toe velocity.
        Uses numerical differentiation for simplicity.
        """
        eps = 1e-6
        J = np.zeros((3, TOTAL_DOF), dtype=np.float64)
        
        _, _, toe_0 = self.get_leg_kinematics(leg_idx)
        
        for i in range(TOTAL_DOF):
            q_save = self.q[i]
            self.q[i] += eps
            _, _, toe_plus = self.get_leg_kinematics(leg_idx)
            self.q[i] = q_save
            
            J[:, i] = (toe_plus - toe_0) / eps
        
        return J
    
    def get_all_kinematics(self) -> Dict[str, Any]:
        """Get complete kinematic state for visualization."""
        cfg = self.cfg
        torso_pos = self.q[0:3]
        roll, pitch, yaw = self.q[3], self.q[4], self.q[5]
        R = rotation_matrix_from_euler(roll, pitch, yaw)
        
        # Torso bounding box corners
        half_l = cfg.torso_length / 2
        half_w = cfg.torso_width / 2
        half_h = cfg.torso_height / 2
        
        corners_local = [
            np.array([half_l, half_h, half_w]),
            np.array([half_l, half_h, -half_w]),
            np.array([-half_l, half_h, -half_w]),
            np.array([-half_l, half_h, half_w]),
            np.array([half_l, -half_h, half_w]),
            np.array([half_l, -half_h, -half_w]),
            np.array([-half_l, -half_h, -half_w]),
            np.array([-half_l, -half_h, half_w]),
        ]
        
        torso_corners = [torso_pos + R @ c for c in corners_local]
        
        legs = {}
        for leg_idx in range(4):
            hip, knee, toe = self.get_leg_kinematics(leg_idx)
            legs[LEG_NAMES[leg_idx]] = {
                "hip": hip.tolist(),
                "knee": knee.tolist(),
                "toe": toe.tolist()
            }
        
        return {
            "torso_pos": torso_pos.tolist(),
            "torso_rot": [roll, pitch, yaw],
            "torso_corners": [c.tolist() for c in torso_corners],
            "legs": legs
        }
    
    def compute_coupled_mass_matrix(self) -> np.ndarray:
        """
        Compute mass matrix with coupled leg-torso dynamics.
        
        KEY IMPROVEMENT: Leg inertia contributes to torso rotational inertia,
        and joint inertias are higher to prevent unrealistic fast movements.
        """
        cfg = self.cfg
        M = np.zeros(TOTAL_DOF, dtype=np.float64)
        
        total_mass = cfg.torso_mass + 4 * (cfg.thigh_mass + cfg.calf_mass)
        
        # === Torso translation: feels full body mass ===
        M[0] = total_mass
        M[1] = total_mass
        M[2] = total_mass
        
        # === Torso rotation: includes leg inertia contribution ===
        # When legs swing, they create angular momentum that the body must counter
        leg_inertia_contrib = 4 * (
            cfg.thigh_mass * (cfg.thigh_length ** 2) / 3 +
            cfg.calf_mass * (cfg.thigh_length + cfg.calf_length / 2) ** 2
        )
        
        M[3] = cfg.torso_Ixx + leg_inertia_contrib * 0.3   # roll - less affected
        M[4] = cfg.torso_Izz + leg_inertia_contrib * 0.5   # pitch - legs swing in this plane
        M[5] = cfg.torso_Iyy + leg_inertia_contrib * 0.2   # yaw - some effect
        
        # === Joint inertias: HIGHER to slow down limb movements ===
        # This prevents the "disconnected limb" appearance
        for leg in range(4):
            base_idx = TORSO_DOF + leg * 3
            # Abduction (hip sideways) - lower inertia, smaller range
            M[base_idx + JOINT_ABD] = 0.08
            # Hip flexion - higher inertia, big movement
            M[base_idx + JOINT_HIP] = 0.12
            # Knee - medium inertia
            M[base_idx + JOINT_KNEE] = 0.08
        
        return M
    
    def compute_leg_reaction_torques(self, tau_motor: np.ndarray) -> np.ndarray:
        """
        Compute reaction torques on torso from leg motor torques.
        
        Newton's 3rd law: when we apply torque to accelerate a leg,
        an equal and opposite torque acts on the body.
        """
        tau_react = np.zeros(TOTAL_DOF, dtype=np.float64)
        cfg = self.cfg
        
        for leg_idx in range(4):
            base_idx = leg_idx * 3  # Index in tau_motor (which is LEGS_DOF sized)
            
            # Determine which side this leg is on
            is_left = (leg_idx == LEG_FL or leg_idx == LEG_BL)
            is_front = (leg_idx == LEG_FL or leg_idx == LEG_FR)
            
            # Hip flexion torque creates pitch reaction on body
            hip_torque = tau_motor[base_idx + JOINT_HIP]
            # Scale based on leg position (front legs have more lever arm for pitch)
            pitch_scale = 0.08 if is_front else 0.06
            tau_react[4] -= pitch_scale * hip_torque
            
            # Abduction torque creates roll reaction
            abd_torque = tau_motor[base_idx + JOINT_ABD]
            roll_sign = 1.0 if is_left else -1.0
            tau_react[3] -= roll_sign * 0.05 * abd_torque
            
            # Knee torque also affects pitch slightly
            knee_torque = tau_motor[base_idx + JOINT_KNEE]
            tau_react[4] -= 0.03 * knee_torque
        
        return tau_react
    
    def apply_soft_joint_limits(self) -> None:
        """
        Apply joint limits with soft damping near boundaries.
        
        KEY IMPROVEMENT: Instead of hard stops that cause jerky bouncing,
        we apply increasing damping as joints approach their limits.
        """
        for leg in range(4):
            base_idx = TORSO_DOF + leg * 3
            
            # === Abduction: [-0.5, 0.5] with soft limits ===
            abd_idx = base_idx + JOINT_ABD
            abd_limit = 0.5
            abd_soft_zone = 0.4  # Start damping at 80% of limit
            
            if abs(self.q[abd_idx]) > abd_soft_zone:
                # Apply damping proportional to how far into soft zone
                overshoot = (abs(self.q[abd_idx]) - abd_soft_zone) / (abd_limit - abd_soft_zone)
                overshoot = min(1.0, overshoot)
                damping_factor = 1.0 - 0.5 * overshoot
                self.qd[abd_idx] *= damping_factor
            
            if self.q[abd_idx] > abd_limit:
                self.q[abd_idx] = abd_limit
                self.qd[abd_idx] = min(0, self.qd[abd_idx])
            elif self.q[abd_idx] < -abd_limit:
                self.q[abd_idx] = -abd_limit
                self.qd[abd_idx] = max(0, self.qd[abd_idx])
            
            # === Hip: [-1.0, 1.0] (reduced from 1.2) ===
            hip_idx = base_idx + JOINT_HIP
            hip_limit = 1.0
            hip_soft_zone = 0.8
            
            if abs(self.q[hip_idx]) > hip_soft_zone:
                overshoot = (abs(self.q[hip_idx]) - hip_soft_zone) / (hip_limit - hip_soft_zone)
                overshoot = min(1.0, overshoot)
                self.qd[hip_idx] *= (1.0 - 0.5 * overshoot)
            
            if self.q[hip_idx] > hip_limit:
                self.q[hip_idx] = hip_limit
                self.qd[hip_idx] = min(0, self.qd[hip_idx])
            elif self.q[hip_idx] < -hip_limit:
                self.q[hip_idx] = -hip_limit
                self.qd[hip_idx] = max(0, self.qd[hip_idx])
            
            # === Knee: [-2.2, -0.15] (must stay bent) ===
            knee_idx = base_idx + JOINT_KNEE
            knee_upper = -0.15
            knee_lower = -2.2
            knee_soft_upper = -0.3
            knee_soft_lower = -2.0
            
            if self.q[knee_idx] > knee_soft_upper:
                overshoot = (self.q[knee_idx] - knee_soft_upper) / (knee_upper - knee_soft_upper)
                overshoot = min(1.0, overshoot)
                self.qd[knee_idx] *= (1.0 - 0.5 * overshoot)
            elif self.q[knee_idx] < knee_soft_lower:
                overshoot = (knee_soft_lower - self.q[knee_idx]) / (knee_soft_lower - knee_lower)
                overshoot = min(1.0, overshoot)
                self.qd[knee_idx] *= (1.0 - 0.5 * overshoot)
            
            if self.q[knee_idx] > knee_upper:
                self.q[knee_idx] = knee_upper
                self.qd[knee_idx] = min(0, self.qd[knee_idx])
            elif self.q[knee_idx] < knee_lower:
                self.q[knee_idx] = knee_lower
                self.qd[knee_idx] = max(0, self.qd[knee_idx])
    
    def step(self, targets: np.ndarray) -> Tuple[List[float], np.ndarray]:
        """
        Advance physics simulation by one timestep.
        
        Args:
            targets: Target joint angles (12,) - directly from CPG + residuals
            
        Returns:
            Tuple of (contact_flags, motor_torques)
        """
        cfg = self.cfg
        targets = np.asarray(targets, dtype=np.float64).reshape(LEGS_DOF,)
        
        # === PD Control to compute motor torques ===
        joint_q = self.q[TORSO_DOF:]
        joint_qd = self.qd[TORSO_DOF:]
        
        tau_motor = self.kp * (targets - joint_q) - self.kd * joint_qd
        tau_motor = np.clip(tau_motor, -TAU_MAX, TAU_MAX)
        
        # === Compute external forces from ground contact ===
        tau_ext = np.zeros(TOTAL_DOF, dtype=np.float64)
        contact_flags = [0.0, 0.0, 0.0, 0.0]
        
        for leg_idx in range(4):
            _, _, toe_pos = self.get_leg_kinematics(leg_idx)
            
            # Compute toe velocity using Jacobian
            J = self.compute_leg_jacobian(leg_idx)
            toe_vel = J @ self.qd
            
            # Ground reaction force
            F_contact = self.floor.compute_force(
                f"{LEG_NAMES[leg_idx]}_toe",
                toe_pos,
                toe_vel
            )
            
            # Map contact force to joint torques (J^T * F)
            tau_ext += J.T @ F_contact
            
            # Record contact for reward computation
            if toe_pos[1] < 0.01:
                contact_flags[leg_idx] = 1.0
        
        # === Compute mass matrix (coupled dynamics) ===
        M = self.compute_coupled_mass_matrix()
        
        # === Assemble total torques ===
        tau = np.zeros(TOTAL_DOF, dtype=np.float64)
        
        # Gravity on torso
        total_mass = cfg.torso_mass + 4 * (cfg.thigh_mass + cfg.calf_mass)
        tau[1] = -total_mass * cfg.g
        
        # Motor torques
        tau[TORSO_DOF:] += tau_motor
        
        # Ground reaction
        tau += tau_ext
        
        # === NEW: Reaction torques from leg movements ===
        tau_react = self.compute_leg_reaction_torques(tau_motor)
        tau += tau_react
        
        # General damping for stability
        tau -= 0.15 * self.qd  # Slightly increased from 0.1
        
        # === Integrate dynamics ===
        qdd = tau / (M + 1e-9)
        self.prev_qdd_joints = qdd[TORSO_DOF:].copy()
        
        self.qd += qdd * cfg.dt
        self.q += self.qd * cfg.dt
        self.t += cfg.dt
        
        # === Apply joint limits with soft damping ===
        self.apply_soft_joint_limits()
        
        # Normalize torso angles
        for i in [3, 4, 5]:
            self.q[i] = normalize_angle(self.q[i])
        
        return contact_flags, tau_motor


# =============================================================================
# CALIBRATION
# =============================================================================

def run_calibration() -> Tuple[float, float]:
    """
    Find good PD gains by testing stability during standing.
    """
    add_log("🔧 Calibrating 3D physics (KP/KD)…")
    best_score = 1e18
    best_params = (DEFAULT_KP, DEFAULT_KD)
    
    kps = [40, 50, 60, 80]
    kds = [2, 3, 4, 6]
    
    cfg = DogConfig3D()
    
    for kp in kps:
        for kd in kds:
            sim = DogModel3D(cfg)
            sim.kp = kp
            sim.kd = kd
            
            sim.q[1] = 0.6  # Start slightly high
            
            total_jerk = 0.0
            for _ in range(500):
                # Stand still - just maintain pose
                targets = np.array([0.0, 0.25, -0.7] * 4, dtype=np.float64)
                sim.step(targets)
                total_jerk += float(np.sum(np.abs(sim.qd)))
            
            # Penalize falling
            if sim.q[1] < 0.2:
                total_jerk += 10000.0
            
            # Penalize tilting
            if abs(sim.q[3]) > 0.5 or abs(sim.q[4]) > 0.5:
                total_jerk += 5000.0
            
            if total_jerk < best_score:
                best_score = total_jerk
                best_params = (float(kp), float(kd))
    
    add_log(f"✅ Calibration complete: Kp={best_params[0]} Kd={best_params[1]}")
    return best_params


# =============================================================================
# GENOME DECODER
# =============================================================================

def _softmax_np(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    x = x.astype(np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)


class GeneDecoder:
    """Decode genome vector into network hyperparameters."""
    
    @staticmethod
    def decode(vec: np.ndarray) -> Dict[str, Any]:
        v = np.clip(np.asarray(vec, dtype=np.float64), 0.0, 1.0)
        
        # Reservoir size
        n_res = int(v[0] * (MAX_RESERVOIR_SIZE - MIN_RESERVOIR_SIZE) + MIN_RESERVOIR_SIZE)
        
        # Connection density
        density = v[1] * (0.40 - MIN_DENSITY) + MIN_DENSITY
        
        # Leak rates for different timescale pools
        leak_fast = v[2] * 0.45 + 0.40   # 0.40 - 0.85
        leak_med  = v[3] * 0.33 + 0.12   # 0.12 - 0.45
        leak_slow = v[4] * 0.13 + 0.02   # 0.02 - 0.15
        
        # Pool sizes
        pool_fast = v[5] * 0.35 + 0.15   # 0.15 - 0.50
        pool_slow = v[6] * 0.35 + 0.15   # 0.15 - 0.50
        if pool_fast + pool_slow > 0.85:
            scale = 0.85 / (pool_fast + pool_slow)
            pool_fast *= scale
            pool_slow *= scale
        pool_med = 1.0 - pool_fast - pool_slow
        
        # Spectral radius
        radius = v[7] * (MAX_SPECTRAL_RADIUS - 0.6) + 0.6
        
        # Learning rate (log scale)
        lr = float(10 ** (-4.2 + (2.0 * v[8])))
        
        # Input gain
        input_gain = 1.8
        
        # Reward weights (evolved)
        raw = (v[9:14] * 4.0) - 2.0
        w = _softmax_np(raw).astype(np.float64)
        
        # Ensure minimum alive weight
        if w[0] < MIN_ALIVE_WEIGHT:
            w[0] = MIN_ALIVE_WEIGHT
            rest = w[1:].sum()
            if rest > 1e-9:
                w[1:] = w[1:] * (1.0 - MIN_ALIVE_WEIGHT) / rest
            else:
                w[1:] = (1.0 - MIN_ALIVE_WEIGHT) / 4.0
        w = w / w.sum()
        
        return {
            "n_reservoir": int(n_res),
            "density": float(density),
            "leak_fast": float(leak_fast),
            "leak_med": float(leak_med),
            "leak_slow": float(leak_slow),
            "pool_fast": float(pool_fast),
            "pool_med": float(pool_med),
            "pool_slow": float(pool_slow),
            "spectral_radius": float(radius),
            "lr": float(lr),
            "input_gain": float(input_gain),
            "reward_w": [float(x) for x in w.tolist()],
        }


# =============================================================================
# 3D ENVIRONMENT - ENHANCED WITH CPG AND COORDINATION REWARDS
# =============================================================================

class DogEnv3D:
    """
    Quadruped locomotion environment with CPG-based control.
    
    KEY IMPROVEMENTS:
    1. CPG provides rhythmic baseline - network learns residuals
    2. Gait coordination rewards encourage proper locomotion patterns
    3. Smoother action transitions penalize jerky movements
    """
    
    def __init__(self, kp: float, kd: float, reward_w: np.ndarray):
        self.cfg = DogConfig3D()
        self.model = DogModel3D(self.cfg)
        self.model.kp = float(kp)
        self.model.kd = float(kd)
        self.reward_w = np.asarray(reward_w, dtype=np.float64).reshape(REWARD_D,)
        
        # Initialize CPG
        self.cpg = CPGPrior(freq=CPG_FREQUENCY, duty_cycle=CPG_DUTY_CYCLE)
        
        self.reset()
    
    def reset(self) -> torch.Tensor:
        """Reset environment to initial state."""
        self.model.reset()
        self.cpg.reset()
        
        # Small random perturbation to initial pose
        self.model.q[0:3] += np.random.uniform(-0.02, 0.02, 3)
        self.model.q[3:6] += np.random.uniform(-0.03, 0.03, 3)
        
        self.steps = 0
        self.contacts = [0.0, 0.0, 0.0, 0.0]
        self.prev_x = float(self.model.q[0])
        self.start_x = float(self.model.q[0])
        self.last_tau_cost = 0.0
        
        # For smoothness and stability rewards
        self.prev_action = np.zeros(ACTION_DIM, dtype=np.float64)
        self.prev_cpg_targets = self.cpg.get_target_angles()
        self.height_history: List[float] = []
        
        return self.get_obs()
    
    def get_obs(self) -> torch.Tensor:
        """
        Construct observation vector.
        
        Enhanced with CPG phase signals so network knows where in gait cycle it is.
        """
        q = self.model.q
        qd = self.model.qd
        
        obs = []
        
        # Height
        obs.append(float(q[1]))
        
        # Torso orientation (sin/cos encoding for continuity)
        obs.append(math.sin(q[3]))  # roll
        obs.append(math.cos(q[3]))
        obs.append(math.sin(q[4]))  # pitch
        obs.append(math.cos(q[4]))
        obs.append(math.sin(q[5]))  # yaw
        obs.append(math.cos(q[5]))
        
        # Joint angles (12 values)
        for i in range(LEGS_DOF):
            obs.append(float(q[TORSO_DOF + i]))
        
        # Torso velocities (6 values)
        obs.append(float(qd[0]))
        obs.append(float(qd[1]))
        obs.append(float(qd[2]))
        obs.append(float(qd[3]))
        obs.append(float(qd[4]))
        obs.append(float(qd[5]))
        
        # Joint velocities (12 values)
        for i in range(LEGS_DOF):
            obs.append(float(qd[TORSO_DOF + i]))
        
        # Contact flags (4 values)
        for c in self.contacts:
            obs.append(float(c))
        
        # Progress signals
        obs.append(float(q[0] - self.prev_x))  # Forward progress
        obs.append(float(q[1] - 0.35))         # Height error
        obs.append(float(q[2]))                 # Lateral position
        
        # === NEW: CPG phase signals (8 values) ===
        phase_encoding = self.cpg.get_phase_encoding()
        for v in phase_encoding:
            obs.append(float(v))
        
        # === NEW: CPG target differences (4 values, one per leg) ===
        # Shows how far current pose is from CPG target
        cpg_targets = self.cpg.get_target_angles()
        current_joints = q[TORSO_DOF:]
        for leg in range(4):
            base = leg * 3
            diff = np.mean(np.abs(current_joints[base:base+3] - cpg_targets[base:base+3]))
            obs.append(float(diff))
        
        # Pad to fixed size
        while len(obs) < OBS_DIM:
            obs.append(0.0)
        
        return torch.tensor(obs[:OBS_DIM], dtype=torch.float32).unsqueeze(0)
    
    def compute_coordination_reward(self, action: np.ndarray, info: Dict[str, float]) -> float:
        """
        Compute gait coordination rewards (fixed, not evolved).
        
        These encourage proper locomotion patterns:
        1. Diagonal gait synchronization
        2. Action symmetry between diagonal pairs
        3. Smooth action transitions
        4. Height stability
        5. Lateral drift penalty
        """
        reward = 0.0
        
        # === 1. DIAGONAL GAIT SYNCHRONIZATION ===
        # Trotting gait: FL+BR and FR+BL should contact together
        c = self.contacts
        diag1_sync = c[0] * c[3]  # FL and BR together
        diag2_sync = c[1] * c[2]  # FR and BL together
        # Penalize same-side contacts (not a proper gait)
        same_side_penalty = (c[0] * c[2] + c[1] * c[3]) * 0.5  # left-left, right-right
        r_gait = COORD_WEIGHT_DIAG_GAIT * (diag1_sync + diag2_sync - same_side_penalty)
        reward += r_gait
        
        # === 2. ACTION SYMMETRY ===
        # Diagonal pairs should have similar actions (with appropriate phase offset)
        action_4x3 = action.reshape(4, 3)  # [leg, joint]
        # FL (0) vs BR (3) - diagonal pair
        sym_diag1 = np.mean(np.abs(action_4x3[0] - action_4x3[3]))
        # FR (1) vs BL (2) - diagonal pair
        sym_diag2 = np.mean(np.abs(action_4x3[1] - action_4x3[2]))
        r_symmetry = -COORD_WEIGHT_SYMMETRY * (sym_diag1 + sym_diag2)
        reward += r_symmetry
        
        # === 3. ACTION SMOOTHNESS ===
        # Penalize jerky changes in action
        action_delta = np.abs(action - self.prev_action)
        r_smooth = -COORD_WEIGHT_SMOOTH * np.mean(action_delta)
        reward += r_smooth
        
        # === 4. HEIGHT STABILITY ===
        # Penalize bobbing up and down
        h = info["height"]
        self.height_history.append(h)
        if len(self.height_history) > 20:
            self.height_history.pop(0)
        if len(self.height_history) >= 5:
            height_var = np.var(self.height_history)
            r_stable = -COORD_WEIGHT_STABLE * height_var
            reward += r_stable
        
        # === 5. LATERAL DRIFT PENALTY ===
        # Encourage walking straight
        r_lateral = -COORD_WEIGHT_LATERAL * abs(info["lateral"])
        reward += r_lateral
        
        return reward
    
    def step(self, action: np.ndarray) -> Tuple[torch.Tensor, float, bool, Dict[str, float]]:
        """
        Execute one environment step.
        
        The action from the network is interpreted as RESIDUALS on top of
        the CPG baseline, not raw joint angles.
        """
        action = np.asarray(action, dtype=np.float64).reshape(ACTION_DIM,)
        
        # === CPG provides baseline targets ===
        cpg_targets = self.cpg.get_target_angles()
        
        # === Network outputs residuals (scaled) ===
        residuals = action * RESIDUAL_SCALE
        
        # === Final targets = CPG baseline + learned residuals ===
        targets = cpg_targets + residuals
        
        # Clip to valid joint ranges
        for leg in range(4):
            base = leg * 3
            targets[base + 0] = np.clip(targets[base + 0], -0.5, 0.5)   # abd
            targets[base + 1] = np.clip(targets[base + 1], -1.0, 1.0)   # hip
            targets[base + 2] = np.clip(targets[base + 2], -2.2, -0.15) # knee
        
        # === Run physics substeps ===
        self.contacts = [0.0, 0.0, 0.0, 0.0]
        tau_cost_acc = 0.0
        
        for _ in range(SUBSTEPS):
            c_flags, tau_motor = self.model.step(targets)
            for i in range(4):
                self.contacts[i] = max(self.contacts[i], float(c_flags[i]))
            tau_cost_acc += float(np.mean((tau_motor / TAU_MAX) ** 2))
            self.cpg.step(DT_PHYS)  # Advance CPG
        
        self.last_tau_cost = tau_cost_acc / max(1, SUBSTEPS)
        
        # === Extract state for reward computation ===
        h = float(self.model.q[1])
        roll = float(self.model.q[3])
        pitch = float(self.model.q[4])
        yaw = float(self.model.q[5])
        
        x = float(self.model.q[0])
        z = float(self.model.q[2])
        dx = x - self.prev_x
        self.prev_x = x
        
        distance = x - self.start_x
        vel = dx / (CONTROL_DT + 1e-9)
        
        # === BASE REWARD (evolved weights) ===
        upright = h > 0.20 and abs(roll) < 0.8 and abs(pitch) < 0.8
        r_alive = 1.0 if upright else 0.0
        
        # Forward velocity reward (main objective)
        r_dist = float(np.clip(vel / 2.0, -1.0, 2.0))
        
        # Upright bonus
        r_up = -(abs(roll) + abs(pitch)) * 0.5
        
        # Energy efficiency (penalize motor effort)
        r_tau = -self.last_tau_cost
        
        # Base contact reward
        r_cnt = 0.25 * sum(self.contacts)
        
        # Combine with evolved weights
        c_vec = np.array([r_alive, r_dist, r_up, r_tau, r_cnt], dtype=np.float64)
        base_reward = float(np.dot(self.reward_w, c_vec))
        
        # === COORDINATION REWARD (fixed weights) ===
        info = {
            "distance": float(distance),
            "steps": float(self.steps),
            "height": float(h),
            "roll": float(roll),
            "pitch": float(pitch),
            "yaw": float(yaw),
            "vel": float(vel),
            "lateral": float(z),
        }
        
        coord_reward = self.compute_coordination_reward(action, info)
        
        # === TOTAL REWARD ===
        reward = float(np.clip(2.2 * base_reward + coord_reward, -6.0, 6.0))
        
        # Update previous action for smoothness computation
        self.prev_action = action.copy()
        self.prev_cpg_targets = cpg_targets.copy()
        
        self.steps += 1
        
        # === TERMINATION CONDITIONS ===
        done = (
            h < 0.15 or              # Fell down
            abs(roll) > 1.2 or       # Rolled over
            abs(pitch) > 1.2 or      # Flipped
            self.steps >= MAX_STEPS  # Episode timeout
        )
        
        return self.get_obs(), reward, done, info


# =============================================================================
# NEURAL NETWORK
# =============================================================================

class MultiTimescaleReservoir(nn.Module):
    """
    Liquid neural network reservoir with multiple timescale pools.
    
    Different pools have different leak rates, allowing the network
    to integrate information over multiple timescales:
    - Fast pool: quick reactions
    - Medium pool: gait rhythm integration
    - Slow pool: long-term adaptation
    """
    
    def __init__(self, input_dim: int, params: Dict[str, Any]):
        super().__init__()
        self.size = int(params["n_reservoir"])
        self.input_dim = int(input_dim)
        
        # Compute pool sizes
        n_fast = int(self.size * params["pool_fast"])
        n_slow = int(self.size * params["pool_slow"])
        n_med = self.size - n_fast - n_slow
        self.pool_sizes = [n_fast, n_med, n_slow]
        self.pool_boundaries = [0, n_fast, n_fast + n_med, self.size]
        
        # Assign leak rates to neurons based on pool
        leak = torch.zeros(self.size, dtype=torch.float32)
        leak[0:n_fast] = float(params["leak_fast"])
        leak[n_fast:n_fast + n_med] = float(params["leak_med"])
        leak[n_fast + n_med:] = float(params["leak_slow"])
        self.register_buffer("leak", leak)
        
        gain = float(params["input_gain"])
        density = float(params["density"])
        radius = float(params["spectral_radius"])
        
        # Input weights (fixed, random)
        self.w_in = nn.Linear(self.input_dim, self.size, bias=False)
        with torch.no_grad():
            in_mask = (torch.rand(self.size, self.input_dim) < 0.6).float()
            in_w = (torch.rand(self.size, self.input_dim) * 2.0 - 1.0) * gain * in_mask
            self.w_in.weight.copy_(in_w)
            self.w_in.weight.requires_grad_(False)
        
        # Recurrent weights (fixed, sparse, scaled to spectral radius)
        mask = (torch.rand(self.size, self.size) < density).float()
        w_rec = (torch.rand(self.size, self.size) * 2.0 - 1.0) * mask
        
        try:
            eig = torch.linalg.eigvals(w_rec)
            max_e = torch.max(torch.abs(eig)).item()
        except Exception:
            max_e = torch.linalg.norm(w_rec, ord=2).item()
        
        if max_e > 1e-6:
            w_rec = w_rec * (radius / max_e)
        
        self.w_rec = nn.Parameter(w_rec, requires_grad=False)
        self.mask = mask
        self.bias = nn.Parameter(torch.zeros(self.size), requires_grad=False)
        
        # Intrinsic plasticity statistics
        self.register_buffer("ip_mean", torch.zeros(self.size))
        self.register_buffer("ip_var", torch.ones(self.size) * 0.1)
        
        # For visualization
        inds = mask.nonzero().tolist()
        stride = max(1, len(inds) // VIZ_MAX_LINKS)
        self.links = inds[::stride][:VIZ_MAX_LINKS]
        self.link_weights = [float(w_rec[i, j].item()) for i, j in self.links]
    
    def forward(self, u: torch.Tensor, h: torch.Tensor, apply_ip: bool = False) -> torch.Tensor:
        """
        Update reservoir state.
        
        Args:
            u: Input tensor
            h: Previous hidden state
            apply_ip: Whether to apply intrinsic plasticity
            
        Returns:
            New hidden state
        """
        inj = self.w_in(u)
        rec = F.linear(h, self.w_rec)
        pre = inj + rec + self.bias
        act = torch.tanh(pre)
        
        # Intrinsic plasticity (homeostatic adaptation)
        if apply_ip and self.training:
            with torch.no_grad():
                batch_mean = act.mean(dim=0)
                self.ip_mean = 0.99 * self.ip_mean + 0.01 * batch_mean
                mean_err = self.ip_mean - IP_TARGET_MEAN
                self.bias.data -= IP_LEARNING_RATE * mean_err
        
        # Leaky integration
        h_new = (1.0 - self.leak) * h + self.leak * act
        return h_new
    
    def get_pool_info(self) -> Dict[str, Any]:
        """Get pool configuration for visualization."""
        return {
            "pool_sizes": self.pool_sizes,
            "pool_boundaries": self.pool_boundaries,
            "leak_rates": [
                float(self.leak[self.pool_boundaries[0]].item()),
                float(self.leak[self.pool_boundaries[1]].item()) if self.pool_sizes[1] > 0 else 0.0,
                float(self.leak[self.pool_boundaries[2]].item()) if self.pool_sizes[2] > 0 else 0.0,
            ],
        }


class LiquidAgent(nn.Module):
    """
    Actor-Critic agent using a liquid neural network.
    
    The agent outputs RESIDUALS on the CPG baseline, not raw actions.
    This is much easier to learn than generating locomotion from scratch.
    """
    
    def __init__(self, params: Dict[str, Any]):
        super().__init__()
        self.params = params
        self.reservoir = MultiTimescaleReservoir(INPUT_DIM, params)
        
        # Actor outputs mean and log_std for Gaussian policy
        self.actor = nn.Linear(self.reservoir.size, ACTION_DIM * 2)
        
        # Critic outputs state value estimate
        self.critic = nn.Linear(self.reservoir.size, 1)
    
    def init_hidden(self, batch: int = 1) -> torch.Tensor:
        """Initialize hidden state."""
        return torch.zeros(batch, self.reservoir.size, dtype=torch.float32)
    
    def forward(self, obs: torch.Tensor, prev_action: torch.Tensor, h: torch.Tensor,
                apply_ip: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through network.
        
        Returns:
            Tuple of (action_mean, action_log_std, value, new_hidden_state)
        """
        u = torch.cat([obs, prev_action], dim=1)
        h2 = self.reservoir(u, h, apply_ip=apply_ip)
        out = self.actor(h2)
        mu = out[:, :ACTION_DIM]
        log_std = torch.clamp(out[:, ACTION_DIM:], -3.0, 1.0)
        v = self.critic(h2).squeeze(1)
        return mu, log_std, v, h2
    
    def sample(self, obs: torch.Tensor, prev_action: torch.Tensor, h: torch.Tensor,
               apply_ip: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Returns:
            Tuple of (action, log_prob, value, entropy, new_hidden_state)
        """
        mu, log_std, v, h2 = self.forward(obs, prev_action, h, apply_ip=apply_ip)
        std = torch.exp(log_std)
        eps = torch.randn_like(mu)
        u = mu + std * eps
        a = torch.tanh(u)  # Squash to [-1, 1]
        
        # Log probability with tanh squashing correction
        normal_logp = -0.5 * (((u - mu) / (std + 1e-8)) ** 2 + 2 * log_std + math.log(2 * math.pi))
        logp = normal_logp.sum(dim=1) - torch.log(1 - a ** 2 + 1e-6).sum(dim=1)
        
        # Entropy
        entropy = (log_std + 0.5 * math.log(2 * math.pi * math.e)).sum(dim=1)
        
        return a, logp, v, entropy, h2
    
    @torch.no_grad()
    def act_deterministic(self, obs: torch.Tensor, prev_action: torch.Tensor, h: torch.Tensor
                          ) -> Tuple[np.ndarray, torch.Tensor, np.ndarray, float]:
        """
        Get deterministic action (mean of policy).
        Used during evaluation and demo.
        """
        mu, _, v, h2 = self.forward(obs, prev_action, h, apply_ip=False)
        a = torch.tanh(mu)
        return a.squeeze(0).cpu().numpy(), h2, mu.squeeze(0).cpu().numpy(), float(v.item())
    
    @torch.no_grad()
    def get_activations(self, h: torch.Tensor) -> List[float]:
        """Get reservoir activations for visualization."""
        h_np = h.squeeze(0).cpu().numpy()
        stride = max(1, len(h_np) // VIZ_MAX_NEURONS)
        return h_np[::stride][:VIZ_MAX_NEURONS].tolist()


# =============================================================================
# RETURNS COMPUTATION
# =============================================================================

def _discounted_returns(rewards: List[float], gamma: float) -> torch.Tensor:
    """Compute discounted returns with normalization."""
    R = 0.0
    out: List[float] = []
    for r in reversed(rewards):
        R = r + gamma * R
        out.insert(0, R)
    t = torch.tensor(out, dtype=torch.float32)
    if len(t) > 1:
        t = (t - t.mean()) / (t.std() + 1e-8)
    return t


# =============================================================================
# WORKER LIFE CYCLE
# =============================================================================

def run_life_cycle(packet: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run training and evaluation for a single genome.
    
    This function runs in a worker process.
    """
    torch.set_num_threads(1)
    
    # Seeding for reproducibility
    seed = int(packet.get("seed", 0) + 1337)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    genome = np.asarray(packet["genome"], dtype=np.float64)
    gen = int(packet["gen"])
    pretrained = packet.get("weights", None)
    kp = float(packet["kp"])
    kd = float(packet["kd"])
    
    try:
        # Decode genome into network parameters
        decoded = GeneDecoder.decode(genome)
        reward_w = np.asarray(decoded["reward_w"], dtype=np.float64)
        
        # Create environment and agent
        env = DogEnv3D(kp=kp, kd=kd, reward_w=reward_w)
        agent = LiquidAgent(decoded).to(DEVICE)
        
        # Try to load pretrained weights
        titan = False
        if pretrained is not None:
            try:
                agent.load_state_dict(pretrained, strict=True)
                titan = True
            except Exception:
                pass
        
        # Optimizer for actor and critic only (reservoir is fixed)
        lr = float(decoded["lr"])
        opt = torch.optim.Adam(
            list(agent.actor.parameters()) + list(agent.critic.parameters()),
            lr=lr
        )
        
        def train_block(episodes: int) -> None:
            """Run training for specified number of episodes."""
            agent.train()
            for _ in range(episodes):
                obs = env.reset().to(DEVICE)
                prev_action = torch.zeros(1, ACTION_DIM, dtype=torch.float32).to(DEVICE)
                h = agent.init_hidden(1).to(DEVICE)
                
                logps: List[torch.Tensor] = []
                vals: List[torch.Tensor] = []
                ents: List[torch.Tensor] = []
                rewards: List[float] = []
                
                for _t in range(MAX_STEPS):
                    a_t, logp_t, v_t, ent_t, h = agent.sample(obs, prev_action, h, apply_ip=True)
                    a_np = a_t.squeeze(0).detach().cpu().numpy()
                    
                    obs2, r, done, _info = env.step(a_np)
                    obs = obs2.to(DEVICE)
                    prev_action = a_t.detach()
                    
                    logps.append(logp_t)
                    vals.append(v_t)
                    ents.append(ent_t)
                    rewards.append(float(r))
                    
                    if done:
                        break
                
                # Compute returns and advantages
                rets = _discounted_returns(rewards, GAMMA).to(DEVICE)
                logps_t = torch.cat(logps)
                vals_t = torch.cat(vals)
                ents_t = torch.cat(ents)
                adv = rets - vals_t.detach()
                
                # A2C loss
                loss = (
                    -(logps_t * adv).sum() +                           # Policy gradient
                    VALUE_COEF * 0.5 * ((vals_t - rets) ** 2).sum() -  # Value loss
                    ENTROPY_COEF * ents_t.sum()                         # Entropy bonus
                )
                
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), GRAD_CLIP)
                opt.step()
        
        # === INITIAL TRAINING ===
        train_block(TRAIN_EPISODES_INITIAL)
        
        # === SCOUT EVALUATION ===
        agent.eval()
        scout_dist = []
        scout_steps = []
        for _ in range(SCOUT_EPISODES):
            obs = env.reset().to(DEVICE)
            prev_action = torch.zeros(1, ACTION_DIM, dtype=torch.float32).to(DEVICE)
            h = agent.init_hidden(1).to(DEVICE)
            steps = 0
            dist = 0.0
            for _t in range(MAX_STEPS):
                a_np, h, _mu, _v = agent.act_deterministic(obs, prev_action, h)
                obs2, _r, done, info = env.step(a_np)
                obs = obs2.to(DEVICE)
                prev_action = torch.from_numpy(a_np).float().unsqueeze(0).to(DEVICE)
                dist = info["distance"]
                steps += 1
                if done:
                    break
            scout_dist.append(dist)
            scout_steps.append(steps)
        
        scout_mean_dist = float(np.mean(scout_dist))
        scout_mean_steps = float(np.mean(scout_steps))
        
        # === VARSITY TRAINING (if promising) ===
        varsity = False
        if scout_mean_dist >= VARSITY_DISTANCE_THRESHOLD:
            varsity = True
            train_block(TRAIN_EPISODES_VARSITY)
        
        # === FINAL EVALUATION ===
        agent.eval()
        eval_dist, eval_steps, eval_return = [], [], []
        for _ in range(EVAL_EPISODES):
            obs = env.reset().to(DEVICE)
            prev_action = torch.zeros(1, ACTION_DIM, dtype=torch.float32).to(DEVICE)
            h = agent.init_hidden(1).to(DEVICE)
            total_r = 0.0
            steps = 0
            dist = 0.0
            for _t in range(MAX_STEPS):
                a_np, h, _mu, _v = agent.act_deterministic(obs, prev_action, h)
                obs2, r, done, info = env.step(a_np)
                obs = obs2.to(DEVICE)
                prev_action = torch.from_numpy(a_np).float().unsqueeze(0).to(DEVICE)
                total_r += float(r)
                dist = info["distance"]
                steps += 1
                if done:
                    break
            eval_dist.append(dist)
            eval_steps.append(steps)
            eval_return.append(total_r)
        
        mean_distance = float(np.mean(eval_dist))
        mean_steps = float(np.mean(eval_steps))
        mean_return = float(np.mean(eval_return))
        
        # Save weights if reasonably good
        trapped = None
        if mean_distance >= max(0.5 * VARSITY_DISTANCE_THRESHOLD, 0.2):
            trapped = {k: v.detach().cpu() for k, v in agent.state_dict().items()}
        
        return {
            "mean_distance": mean_distance,
            "mean_steps": mean_steps,
            "mean_return": mean_return,
            "scout_mean_distance": scout_mean_dist,
            "scout_mean_steps": scout_mean_steps,
            "genome": genome.tolist(),
            "params": decoded,
            "id": f"G{gen}-{random.randint(100,999)}",
            "varsity": varsity,
            "titan": titan,
            "weights": trapped,
        }
    
    except Exception as e:
        import traceback
        return {
            "mean_distance": -1e9,
            "error": str(e) + "\n" + traceback.format_exc(),
            "genome": genome.tolist()
        }


# =============================================================================
# EVOLUTION ENGINE
# =============================================================================

class NaturalGradientEvolution:
    """
    Evolutionary optimizer with adaptive per-gene mutation rates.
    
    Tracks which genes are sensitive to changes (affect fitness) and
    reduces mutation there while increasing exploration of less sensitive genes.
    """
    
    def __init__(self, genome_dim: int):
        self.dim = int(genome_dim)
        self.sigma = np.ones(self.dim, dtype=np.float64) * INIT_SIGMA
        self.sensitivity = np.ones(self.dim, dtype=np.float64) * 0.5
        self.stagnation_count = 0
        self.last_best = -1e9
    
    def mutate(self, parent: np.ndarray) -> np.ndarray:
        """Mutate a genome with current sigma values."""
        noise = np.random.randn(self.dim) * self.sigma
        child = np.clip(parent + noise, 0.0, 1.0)
        return child
    
    def crossover_blx(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """BLX-alpha crossover between two parents."""
        child = np.zeros(self.dim, dtype=np.float64)
        for i in range(self.dim):
            lo, hi = (p1[i], p2[i]) if p1[i] < p2[i] else (p2[i], p1[i])
            d = hi - lo
            a = BLX_ALPHA * d
            child[i] = np.random.uniform(lo - a, hi + a)
        return np.clip(child, 0.0, 1.0)
    
    def update_from_generation(self, results: List[Dict[str, Any]]) -> None:
        """Update mutation rates based on generation results."""
        if len(results) < 4:
            return
        
        # Split into elites and rest
        results_sorted = sorted(results, key=lambda r: r["mean_distance"], reverse=True)
        n_elite = max(2, len(results) // 4)
        elites = results_sorted[:n_elite]
        others = results_sorted[n_elite:]
        
        if len(others) < 2:
            return
        
        # Compute gene differences between elites and others
        elite_genomes = np.array([r["genome"] for r in elites], dtype=np.float64)
        other_genomes = np.array([r["genome"] for r in others], dtype=np.float64)
        
        elite_mean = elite_genomes.mean(axis=0)
        other_mean = other_genomes.mean(axis=0)
        gene_diff = np.abs(elite_mean - other_mean)
        
        # Variance across all genomes
        all_genomes = np.array([r["genome"] for r in results], dtype=np.float64)
        gene_var = all_genomes.var(axis=0) + 1e-6
        
        # Sensitivity = how much gene difference correlates with fitness
        raw_sens = gene_diff / (np.sqrt(gene_var) + 1e-6)
        raw_sens = raw_sens / (raw_sens.max() + 1e-6)
        
        # Smooth update
        self.sensitivity = SENSITIVITY_SMOOTHING * self.sensitivity + (1 - SENSITIVITY_SMOOTHING) * raw_sens
        self.sensitivity = np.clip(self.sensitivity, SENSITIVITY_FLOOR, 1.0)
        
        # Adjust sigma: shrink for sensitive genes, grow for insensitive
        for i in range(self.dim):
            if self.sensitivity[i] > SENSITIVITY_THRESHOLD:
                self.sigma[i] *= SIGMA_SHRINK_RATE
            else:
                self.sigma[i] *= SIGMA_GROW_RATE
        self.sigma = np.clip(self.sigma, SIGMA_MIN, SIGMA_MAX)
    
    def check_stagnation(self, current_best: float) -> None:
        """Check for stagnation and boost exploration if needed."""
        if current_best > self.last_best + 0.05:
            self.stagnation_count = 0
            self.last_best = current_best
        else:
            self.stagnation_count += 1
            if self.stagnation_count >= STAGNATION_THRESHOLD:
                self.sigma = np.clip(self.sigma * STAGNATION_SIGMA_BOOST, SIGMA_MIN, SIGMA_MAX)
                add_log(f"⚡ Stagnation ({STAGNATION_THRESHOLD} gens) — boosting σ")
                self.stagnation_count = 0
    
    def reproduce(self, elites: List[Dict[str, Any]], pop_size: int, generation: int) -> List[Dict[str, Any]]:
        """Create new population from elites."""
        new_pop: List[Dict[str, Any]] = []
        
        # Keep elites unchanged
        for e in elites:
            new_pop.append({"genome": np.array(e["genome"], dtype=np.float64), "weights": e.get("weights")})
        
        # Inject random individuals periodically
        inject = (generation % DIVERSITY_INJECT_EVERY == 0) or (self.stagnation_count > STAGNATION_THRESHOLD // 2)
        n_random = DIVERSITY_INJECT_COUNT if inject else 0
        if n_random > 0:
            add_log(f"🌱 Injecting {n_random} random individuals")
        
        for _ in range(n_random):
            if len(new_pop) < pop_size:
                new_pop.append({"genome": np.random.rand(self.dim), "weights": None})
        
        # Fill rest with mutations and crossovers
        while len(new_pop) < pop_size:
            if random.random() < CROSSOVER_PROB and len(elites) >= 2:
                p1, p2 = random.sample(elites, 2)
                g = self.crossover_blx(np.array(p1["genome"]), np.array(p2["genome"]))
                g = np.clip(g + np.random.randn(self.dim) * self.sigma * 0.5, 0.0, 1.0)
                w = p1.get("weights") if p1.get("mean_distance", -1e9) >= p2.get("mean_distance", -1e9) else p2.get("weights")
                new_pop.append({"genome": g, "weights": w})
            else:
                p = random.choice(elites)
                g = self.mutate(np.array(p["genome"], dtype=np.float64))
                new_pop.append({"genome": g, "weights": p.get("weights")})
        
        return new_pop
    
    def save(self, path: str) -> None:
        """Save evolution state."""
        data = {
            "sigma": self.sigma.tolist(),
            "sensitivity": self.sensitivity.tolist(),
            "stagnation_count": self.stagnation_count,
            "last_best": self.last_best,
        }
        with open(path, "w") as f:
            json.dump(data, f)
    
    def load(self, path: str) -> bool:
        """Load evolution state."""
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self.sigma = np.array(data["sigma"], dtype=np.float64)
            self.sensitivity = np.array(data["sensitivity"], dtype=np.float64)
            self.stagnation_count = int(data.get("stagnation_count", 0))
            self.last_best = float(data.get("last_best", -1e9))
            return True
        except Exception:
            return False


# =============================================================================
# SAVE/LOAD
# =============================================================================

def load_data() -> None:
    """Load saved best genome and weights."""
    if os.path.exists(SAVE_FILE):
        try:
            with open(SAVE_FILE, "r") as f:
                data = json.load(f)
            g = data.get("genome")
            if g and len(g) == GENOME_DIM:
                with _STATE_LOCK:
                    SYSTEM_STATE["best_genome"] = g
                    SYSTEM_STATE["generation"] = int(data.get("gen", 0))
                    if data.get("metric") == BEST_METRIC_TAG:
                        SYSTEM_STATE["best_distance"] = float(data.get("best_distance", -1e9))
                        SYSTEM_STATE["best_steps"] = float(data.get("best_steps", 0.0))
                        add_log(f"📂 Loaded best distance: {SYSTEM_STATE['best_distance']:.3f}")
                    else:
                        add_log("📂 Loaded genome, metric mismatch — reset score")
        except Exception as e:
            add_log(f"⚠️ Save load error: {e}")
    
    if os.path.exists(WEIGHTS_FILE):
        try:
            with _STATE_LOCK:
                SYSTEM_STATE["best_weights"] = torch.load(WEIGHTS_FILE, map_location=DEVICE)
            add_log("🧠 Loaded champion weights")
        except Exception as e:
            add_log(f"⚠️ Weight load error: {e}")


def save_data(best_distance: float, best_steps: float, gen: int, genome: List[float], weights: Optional[dict]) -> None:
    """Save best genome and weights."""
    with open(SAVE_FILE, "w") as f:
        json.dump({
            "metric": BEST_METRIC_TAG,
            "best_distance": best_distance,
            "best_steps": best_steps,
            "gen": gen,
            "genome": genome
        }, f)
    if weights is not None:
        try:
            torch.save(weights, WEIGHTS_FILE)
        except Exception as e:
            add_log(f"⚠️ Weight save error: {e}")


# =============================================================================
# EVOLUTION ENGINE THREAD
# =============================================================================

class EvolutionEngine(threading.Thread):
    """Main evolution loop running in background thread."""
    
    def __init__(self):
        super().__init__(daemon=True)
        self.running = True
        
        # Calibrate physics
        kp, kd = run_calibration()
        with _STATE_LOCK:
            SYSTEM_STATE["kp"] = kp
            SYSTEM_STATE["kd"] = kd
        self.kp = kp
        self.kd = kd
        
        # Initialize population
        self.pop_size = POP_SIZE
        self.population_data = [{"genome": np.random.rand(GENOME_DIM), "weights": None}
                                for _ in range(self.pop_size)]
        
        # Initialize evolution engine
        self.nge = NaturalGradientEvolution(GENOME_DIM)
        if self.nge.load(SIGMA_FILE):
            add_log("📂 Loaded adaptive sigma state")
    
    def _init_default_view(self) -> None:
        """Initialize sim_view with a default standing pose for visualization."""
        # Create a temporary model just to get kinematics
        cfg = DogConfig3D()
        model = DogModel3D(cfg)
        model.reset()
        
        # Get the kinematics in standing pose
        kin = model.get_all_kinematics()
        
        with _STATE_LOCK:
            SYSTEM_STATE["sim_view"] = {
                "t": 0.0,
                "q": model.q.tolist(),
                "qd": model.qd.tolist(),
                "r": 0.0,
                "distance": 0.0,
                "vel": 0.0,
                "height": float(model.q[1]),
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "lateral": 0.0,
                "kin": kin,
                "shadows": {},
                "act": [0.0] * ACTION_DIM,
                "steps": 0,
                "contacts": [0.0, 0.0, 0.0, 0.0],
                "cpg_targets": [0.0] * 12,
                "cpg_phase": [0.0] * 8,
            }
        add_log("🦿 Initialized default standing pose")
    
    def run_demo_mode(self, agent_data: Dict[str, Any]) -> None:
        """Run visualization demo with specified agent."""
        add_log("🎬 Starting demo mode...")
        
        try:
            with _STATE_LOCK:
                SYSTEM_STATE["mode"] = "DEMO"
                SYSTEM_STATE["demo_stop"] = False
                SYSTEM_STATE["demo_resets"] = 0
                SYSTEM_STATE["current_id"] = agent_data.get("id", "CHAMPION")
                SYSTEM_STATE["status"] = "👁️ WATCHING"
            
            params = agent_data["params"]
            weights = agent_data.get("weights")
            
            add_log(f"🧠 Creating env with reservoir={params.get('n_reservoir', '?')}")
            env = DogEnv3D(kp=self.kp, kd=self.kd, reward_w=np.asarray(params["reward_w"], dtype=np.float64))
            agent = LiquidAgent(params).to(DEVICE)
            
            loaded = False
            if weights is not None:
                try:
                    agent.load_state_dict(weights, strict=True)
                    loaded = True
                    add_log("✅ Loaded provided weights")
                except Exception as e:
                    add_log(f"⚠️ Failed to load provided weights: {str(e)[:50]}")
            
            if not loaded:
                with _STATE_LOCK:
                    bw = SYSTEM_STATE.get("best_weights")
                if bw is not None:
                    try:
                        agent.load_state_dict(bw, strict=True)
                        loaded = True
                        add_log("✅ Loaded best weights from state")
                    except Exception as e:
                        add_log(f"⚠️ Failed to load best weights: {str(e)[:50]}")
            
            if not loaded:
                add_log("⚠️ Demo running with RANDOM weights (CPG will still work)")
            
            agent.eval()
            obs = env.reset().to(DEVICE)
            prev_action = torch.zeros(1, ACTION_DIM, dtype=torch.float32).to(DEVICE)
            h = agent.init_hidden(1).to(DEVICE)
            
            add_log(f"🎮 Demo loop starting... obs shape={obs.shape}")
            
            dt_target = 1.0 / 50.0  # 50 FPS
            resets, steps_in_ep = 0, 0
            
            while True:
                t0 = time.time()
                with _STATE_LOCK:
                    if SYSTEM_STATE["mode"] != "DEMO" or SYSTEM_STATE["demo_stop"]:
                        add_log("🛑 Demo stopped by state change")
                        break
                
                try:
                    a_np, h, mu_np, v_est = agent.act_deterministic(obs, prev_action, h)
                    obs2, r, done, info = env.step(a_np)
                    obs = obs2.to(DEVICE)
                    prev_action = torch.from_numpy(a_np).float().unsqueeze(0).to(DEVICE)
                    steps_in_ep += 1
                    
                    kin = env.model.get_all_kinematics()
                    activations = agent.get_activations(h)
                    pool_info = agent.reservoir.get_pool_info()
                    
                    # Get CPG info for visualization
                    cpg_targets = env.cpg.get_target_angles()
                    cpg_phase = env.cpg.get_phase_encoding()
                    
                    with _STATE_LOCK:
                        SYSTEM_STATE["sim_view"] = {
                            "t": float(env.model.t),
                            "q": env.model.q.tolist(),
                            "qd": env.model.qd.tolist(),
                            "r": float(r),
                            "distance": float(info["distance"]),
                            "vel": float(info["vel"]),
                            "height": float(info["height"]),
                            "roll": float(info["roll"]),
                            "pitch": float(info["pitch"]),
                            "yaw": float(info["yaw"]),
                            "lateral": float(info["lateral"]),
                            "kin": kin,
                            "shadows": {k: v.tolist() for k, v in env.model.floor.shadows.items()},
                            "act": [float(x) for x in a_np.tolist()],
                            "steps": int(steps_in_ep),
                            "contacts": env.contacts,
                            "cpg_targets": cpg_targets.tolist(),
                            "cpg_phase": cpg_phase.tolist(),
                        }
                        SYSTEM_STATE["brain_view"] = {
                            "mu": mu_np.tolist(),
                            "v": float(v_est),
                            "links": agent.reservoir.links,
                            "link_weights": agent.reservoir.link_weights,
                            "activations": activations,
                            "n_reservoir": int(agent.reservoir.size),
                            "obs": obs.squeeze(0).cpu().numpy().tolist(),
                            "prev_action": prev_action.squeeze(0).cpu().numpy().tolist(),
                            "pool_info": pool_info,
                        }
                        SYSTEM_STATE["gene_sigma"] = self.nge.sigma.tolist()
                        SYSTEM_STATE["gene_sensitivity"] = self.nge.sensitivity.tolist()
                        SYSTEM_STATE["stagnation_count"] = self.nge.stagnation_count
                    
                    if done:
                        resets += 1
                        with _STATE_LOCK:
                            SYSTEM_STATE["demo_resets"] = int(resets)
                        if resets % 5 == 0:
                            add_log(f"👁️ Demo reset x{resets} (dist={info['distance']:.2f}, steps={steps_in_ep})")
                        obs = env.reset().to(DEVICE)
                        prev_action = torch.zeros(1, ACTION_DIM, dtype=torch.float32).to(DEVICE)
                        h = agent.init_hidden(1).to(DEVICE)
                        steps_in_ep = 0
                    
                    time.sleep(max(0.0, dt_target - (time.time() - t0)))
                    
                except Exception as e:
                    import traceback
                    add_log(f"❌ Demo step error: {str(e)[:60]}")
                    add_log(traceback.format_exc()[:200])
                    break
            
        except Exception as e:
            import traceback
            add_log(f"❌ Demo setup error: {str(e)[:80]}")
            add_log(traceback.format_exc()[:300])
        
        finally:
            with _STATE_LOCK:
                SYSTEM_STATE["mode"] = "TRAINING"
                SYSTEM_STATE["demo_stop"] = False
                SYSTEM_STATE["status"] = "🔥 BACK TO FORGING"
            add_log("🎬 Demo mode ended")
    
    def run(self) -> None:
        """Main evolution loop."""
        load_data()
        gen = 1
        with _STATE_LOCK:
            if SYSTEM_STATE["generation"] > 0:
                gen = SYSTEM_STATE["generation"] + 1
            bg = SYSTEM_STATE.get("best_genome")
            bw = SYSTEM_STATE.get("best_weights")
        
        if bg and len(bg) == GENOME_DIM:
            self.population_data[0] = {"genome": np.array(bg, dtype=np.float64), "weights": bw}
            add_log("🧬 Injected saved champion")
        
        add_log(f"🧪 Evolution: pop={self.pop_size} workers={CORE_COUNT} elites={N_ELITES}")
        add_log(f"📐 3D Model: {TOTAL_DOF} DOF, obs={OBS_DIM}, act={ACTION_DIM}")
        add_log(f"🎛️ CPG: {CPG_FREQUENCY}Hz, residual_scale={RESIDUAL_SCALE}")
        add_log("🎯 Metric: mean_distance (tie-break mean_steps)")
        add_log(f"🌐 UI ready")
        
        # Initialize sim_view with a default standing pose so 3D view shows something
        self._init_default_view()
        
        with ProcessPoolExecutor(max_workers=CORE_COUNT) as executor:
            while self.running:
                # Check for manual demo request
                with _STATE_LOCK:
                    manual = bool(SYSTEM_STATE["manual_demo_request"])
                    if manual:
                        SYSTEM_STATE["manual_demo_request"] = False
                        cg = SYSTEM_STATE.get("best_genome")
                        cw = SYSTEM_STATE.get("best_weights")
                        bd = SYSTEM_STATE.get("best_distance", -1e9)
                        bs = SYSTEM_STATE.get("best_steps", 0.0)
                
                if manual and cg:
                    add_log("▶️ User requested demo")
                    self.run_demo_mode({
                        "genome": cg,
                        "params": GeneDecoder.decode(np.array(cg, dtype=np.float64)),
                        "weights": cw,
                        "id": "CHAMPION",
                        "mean_distance": bd,
                        "mean_steps": bs,
                    })
                elif manual and not cg:
                    add_log("❌ Demo requested but no genome available!")
                
                # Wait if in demo mode
                while True:
                    with _STATE_LOCK:
                        if SYSTEM_STATE["mode"] != "DEMO":
                            break
                    time.sleep(0.25)
                
                # Update status
                with _STATE_LOCK:
                    SYSTEM_STATE["status"] = f"🔥 FORGING GEN {gen}"
                    SYSTEM_STATE["generation"] = int(gen)
                    SYSTEM_STATE["gene_sigma"] = self.nge.sigma.tolist()
                    SYSTEM_STATE["gene_sensitivity"] = self.nge.sensitivity.tolist()
                    SYSTEM_STATE["stagnation_count"] = int(self.nge.stagnation_count)
                
                # Submit all individuals to worker pool
                seed_base = random.randint(1, 1_000_000)
                futures = []
                for i, pd in enumerate(self.population_data):
                    futures.append(executor.submit(run_life_cycle, {
                        "genome": pd["genome"].tolist(),
                        "weights": pd.get("weights"),
                        "gen": gen,
                        "seed": seed_base + i * 17,
                        "kp": self.kp,
                        "kd": self.kd,
                    }))
                
                # Collect results - but check for demo requests periodically
                results: List[Dict[str, Any]] = []
                pending = set(futures)
                total_workers = len(futures)
                last_progress_log = time.time()
                
                while pending:
                    # Check for demo request every 0.5 seconds
                    done_futures = set()
                    for f in list(pending):
                        if f.done():
                            done_futures.add(f)
                            try:
                                res = f.result()
                                if "error" in res:
                                    add_log(f"⚠️ Worker error: {res['error'][:90]}")
                                else:
                                    results.append(res)
                            except Exception as e:
                                add_log(f"⚠️ Worker exception: {str(e)[:60]}")
                    
                    pending -= done_futures
                    
                    # Update status with progress
                    completed = total_workers - len(pending)
                    with _STATE_LOCK:
                        SYSTEM_STATE["status"] = f"🔥 GEN {gen}: {completed}/{total_workers} done"
                    
                    # Log progress every 10 seconds
                    if time.time() - last_progress_log > 10.0 and pending:
                        add_log(f"⏳ Gen {gen}: {completed}/{total_workers} workers complete...")
                        last_progress_log = time.time()
                    
                    if pending:
                        # Check for manual demo request while waiting
                        manual_request = False
                        cg = None
                        cw = None
                        bd = -1e9
                        bs = 0.0
                        
                        with _STATE_LOCK:
                            manual_request = SYSTEM_STATE["manual_demo_request"]
                            if manual_request:
                                SYSTEM_STATE["manual_demo_request"] = False
                                cg = SYSTEM_STATE.get("best_genome")
                                cw = SYSTEM_STATE.get("best_weights")
                                bd = SYSTEM_STATE.get("best_distance", -1e9)
                                bs = SYSTEM_STATE.get("best_steps", 0.0)
                                add_log(f"🔍 Mid-gen demo check: manual={manual_request}, has_genome={cg is not None}")
                        
                        # Run demo OUTSIDE the lock (so UI can update)
                        if manual_request and cg:
                            add_log(f"▶️ Demo requested mid-gen ({len(pending)} workers still running)")
                            self.run_demo_mode({
                                "genome": cg,
                                "params": GeneDecoder.decode(np.array(cg, dtype=np.float64)),
                                "weights": cw,
                                "id": "CHAMPION",
                                "mean_distance": bd,
                                "mean_steps": bs,
                            })
                        elif manual_request and not cg:
                            add_log("❌ Mid-gen demo: no genome!")
                        
                        time.sleep(0.3)  # Short sleep before checking again
                
                if not results:
                    add_log("❌ All workers failed. Reseeding.")
                    self.population_data = [{"genome": np.random.rand(GENOME_DIM), "weights": None}
                                            for _ in range(self.pop_size)]
                    continue
                
                # Update evolution state
                self.nge.update_from_generation(results)
                
                # Sort by fitness
                results.sort(key=lambda r: (r["mean_distance"], r["mean_steps"]), reverse=True)
                elites = results[:N_ELITES]
                best = results[0]
                
                self.nge.check_stagnation(float(best["mean_distance"]))
                
                # Update visualization data
                pop_vecs = [list(r["genome"][:HYPER_D]) + [1.0 if i < N_ELITES else 0.0]
                            for i, r in enumerate(results)]
                avg_elite = np.mean([np.array(e["genome"][:HYPER_D], dtype=np.float64) for e in elites], axis=0).tolist()
                
                with _STATE_LOCK:
                    SYSTEM_STATE["pop_vectors"] = pop_vecs
                    SYSTEM_STATE["history_vectors"].append(avg_elite)
                    if len(SYSTEM_STATE["history_vectors"]) > 160:
                        SYSTEM_STATE["history_vectors"].pop(0)
                    SYSTEM_STATE["hyperparams"] = best["params"]
                    SYSTEM_STATE["current_id"] = best["id"]
                    prior_best = float(SYSTEM_STATE["best_distance"])
                
                # Check for new best
                if float(best["mean_distance"]) > prior_best + 1e-6:
                    with _STATE_LOCK:
                        SYSTEM_STATE["best_distance"] = float(best["mean_distance"])
                        SYSTEM_STATE["best_steps"] = float(best["mean_steps"])
                        SYSTEM_STATE["best_genome"] = best["genome"]
                        if best.get("weights") is not None:
                            SYSTEM_STATE["best_weights"] = best["weights"]
                    
                    save_data(float(best["mean_distance"]), float(best["mean_steps"]), gen, best["genome"], best.get("weights"))
                    self.nge.save(SIGMA_FILE)
                    
                    pools = f"F{best['params']['pool_fast']:.0%}/M{best['params']['pool_med']:.0%}/S{best['params']['pool_slow']:.0%}"
                    alive_w = best["params"]["reward_w"][0]
                    add_log(f"🏆 NEW BEST: {best['id']} dist={best['mean_distance']:.3f} steps={best['mean_steps']:.1f} alive_w={alive_w:.2f} {pools}")
                else:
                    tag = " [T]" if best.get("titan") else (" [V]" if best.get("varsity") else "")
                    add_log(f"Gen {gen}: {best['id']}{tag} dist={best['mean_distance']:.3f} steps={best['mean_steps']:.1f} stag={self.nge.stagnation_count}")
                
                # Auto-demo if really good
                if float(best["mean_distance"]) >= 1.5:
                    add_log("👁️ Auto-demo triggered (distance threshold)")
                    self.run_demo_mode(best)
                
                # Create next generation
                self.population_data = self.nge.reproduce(elites, self.pop_size, gen)
                gen += 1


# =============================================================================
# THREE.JS HTML VISUALIZATION (unchanged from v6)
# =============================================================================

HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>🦿 Dog Lab v7 3D — Iron Wolf (CPG Enhanced)</title>
  <style>
    :root{ --bg:#05070c; --panel:#0b0f14; --panel2:#0f1720; --stroke:#223044; --text:#e6eef8; --muted:#b8c7dd; --c1:#ff9d00; --c2:#ff5500; --ng:#22d3ee; --cross:#a855f7; --warn:#fbbf24; }
    *{box-sizing:border-box;}
    body{margin:0;background:var(--bg);color:var(--text);font-family:ui-sans-serif,system-ui,sans-serif;overflow:hidden;}
    .wrap{display:flex;height:100vh;}
    aside{width:380px;background:var(--panel);border-right:1px solid var(--stroke);padding:14px;display:flex;flex-direction:column;gap:10px;overflow-y:auto;}
    main{flex:1;display:flex;flex-direction:column;}
    .view3d{flex:2;position:relative;}
    .view3d canvas{display:block;}
    .bottom-row{flex:1;display:flex;gap:2px;background:var(--panel);}
    .bottom-row>div{flex:1;position:relative;}
    h1{font-size:15px;margin:0;}
    .card{background:var(--panel2);border:1px solid var(--stroke);border-radius:10px;padding:10px;}
    .row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;}
    button{padding:8px 12px;border-radius:8px;border:1px solid var(--stroke);background:#132034;color:var(--text);cursor:pointer;font-size:12px;}
    button:hover{background:#182a45;}
    button.active{background:#2d4a6f;border-color:#4a7ab0;}
    .stat{display:flex;justify-content:space-between;font-size:11px;color:var(--muted);margin-bottom:4px;}
    .val{color:var(--text);font-weight:700;}
    .mono{font-family:ui-monospace,monospace;font-size:10px;white-space:pre-wrap;}
    canvas.overlay{width:100%;height:100%;display:block;background:var(--panel2);}
    .label{position:absolute;top:6px;left:10px;font-size:10px;color:#7f97b6;z-index:10;pointer-events:none;}
    .tag{display:inline-block;padding:2px 5px;border-radius:4px;font-size:9px;margin-right:3px;}
    .tag-fast{background:#ef4444;color:white;}
    .tag-med{background:#f59e0b;color:black;}
    .tag-slow{background:#3b82f6;color:white;}
    .help{position:absolute;bottom:8px;right:10px;font-size:10px;color:#556;z-index:10;}
    .cpg-indicator{display:flex;gap:4px;margin-top:4px;}
    .cpg-leg{width:20px;height:20px;border-radius:4px;display:flex;align-items:center;justify-content:center;font-size:8px;font-weight:bold;}
    .cpg-swing{background:#ef4444;}
    .cpg-stance{background:#22c55e;}
  </style>
</head>
<body>
<div class="wrap">
  <aside>
    <h1>🦿 Dog Lab v7 3D — Iron Wolf (CPG Enhanced)</h1>
    <div class="card">
      <div class="stat"><span>STATUS</span><span class="val" id="status">—</span></div>
      <div class="stat"><span>MODE</span><span class="val" id="mode">—</span></div>
      <div class="stat"><span>GEN</span><span class="val" id="gen">0</span></div>
      <div class="stat"><span>BEST DIST</span><span class="val" id="best" style="color:var(--c2)">—</span></div>
      <div class="stat"><span>BEST STEPS</span><span class="val" id="bestSteps">—</span></div>
      <div class="stat"><span>STAGNATION</span><span class="val" id="stag">0</span></div>
      <div class="stat"><span>CURRENT</span><span class="val" id="cid">—</span></div>
    </div>
    <div class="card">
      <div class="stat"><span>HEIGHT</span><span class="val" id="hval">—</span></div>
      <div class="stat"><span>ROLL/PITCH</span><span class="val" id="rpval">—</span></div>
      <div class="stat"><span>DISTANCE</span><span class="val" id="dval">—</span></div>
      <div class="stat"><span>VELOCITY</span><span class="val" id="vval">—</span></div>
      <div class="stat"><span>CONTACTS</span><span class="val" id="cval">—</span></div>
      <div class="stat"><span>STEP</span><span class="val" id="sval">—</span></div>
      <div class="stat"><span>CPG PHASE</span><span class="val" id="cpgPhase">—</span></div>
    </div>
    <div class="card">
      <div class="row">
        <button onclick="triggerDemo()">▶ Demo</button>
        <button onclick="directDemo()">⚡ Direct</button>
        <button onclick="stopDemo()">⏹ Stop</button>
        <button onclick="recenter()">⟲ Recenter</button>
      </div>
      <div style="height:6px"></div>
      <div class="row">
        <button id="btnHyper" class="active" onclick="setViz('hyperspace')">🌌 Hyper</button>
        <button id="btnNeural" onclick="setViz('neural')">🧠 Neural</button>
        <button id="btnSigma" onclick="setViz('sigma')">📊 Sigma</button>
      </div>
    </div>
    <div class="card">
      <div class="mono" id="hp">hyperparams…</div>
      <div style="height:4px"></div>
      <div class="mono" id="rw">reward weights…</div>
    </div>
    <div class="card">
      <div style="font-size:11px;color:var(--muted);margin-bottom:4px;"><b>Pools</b></div>
      <div id="pools"></div>
    </div>
    <div class="card">
      <div style="font-size:11px;color:var(--muted);margin-bottom:4px;"><b>Logs</b></div>
      <div class="mono" id="logs" style="max-height:140px;overflow-y:auto;">…</div>
    </div>
    <div class="card" style="background:#1a1020;border-color:#442244;">
      <div style="font-size:10px;color:#a88;"><b>Debug</b></div>
      <div class="mono" id="debug" style="font-size:11px;color:#0f0;background:#111;padding:5px;border:1px solid #333;">waiting for first poll...</div>
    </div>
  </aside>
  <main>
    <div class="view3d" id="view3d">
      <div class="label">3D Simulation (drag to rotate, scroll to zoom, R to recenter)</div>
      <div class="help">Mouse: orbit | Scroll: zoom | R: recenter</div>
    </div>
    <div class="bottom-row">
      <div><div class="label" id="vizLabel">Hyperspace</div><canvas class="overlay" id="map"></canvas></div>
      <div><div class="label">Telemetry</div><canvas class="overlay" id="tele"></canvas></div>
    </div>
  </main>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
// === THREE.JS SETUP ===
const container = document.getElementById('view3d');
let WIDTH = container.clientWidth, HEIGHT = container.clientHeight;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0e14);

const camera = new THREE.PerspectiveCamera(50, WIDTH/HEIGHT, 0.1, 100);
camera.position.set(1.5, 1.0, 1.5);
camera.lookAt(0, 0.3, 0);

const renderer = new THREE.WebGLRenderer({antialias: true});
renderer.setSize(WIDTH, HEIGHT);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
container.appendChild(renderer.domElement);

// Lights
const ambient = new THREE.AmbientLight(0xffffff, 0.4);
scene.add(ambient);
const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
dirLight.position.set(5, 10, 5);
scene.add(dirLight);

// Grid floor
const gridHelper = new THREE.GridHelper(20, 40, 0x2a3a4a, 0x1a2a3a);
scene.add(gridHelper);

// Ground plane
const groundGeo = new THREE.PlaneGeometry(20, 20);
const groundMat = new THREE.MeshStandardMaterial({color: 0x0d1520, roughness: 0.9});
const ground = new THREE.Mesh(groundGeo, groundMat);
ground.rotation.x = -Math.PI / 2;
ground.position.y = -0.001;
scene.add(ground);

// Dog parts
const torsoGeo = new THREE.BoxGeometry(0.5, 0.12, 0.2);
const torsoMat = new THREE.MeshStandardMaterial({color: 0x4080ff, metalness: 0.3, roughness: 0.6});
const torso = new THREE.Mesh(torsoGeo, torsoMat);
torso.position.set(0, 0.35, 0);  // Initial standing height
scene.add(torso);

// Leg materials
const thighMat = new THREE.MeshStandardMaterial({color: 0x60a0ff, metalness: 0.2, roughness: 0.7});
const calfMat = new THREE.MeshStandardMaterial({color: 0x80c0ff, metalness: 0.2, roughness: 0.7});
const toeMat = new THREE.MeshStandardMaterial({color: 0xff6060, metalness: 0.1, roughness: 0.8});

// Leg geometry
const segGeo = new THREE.CylinderGeometry(0.02, 0.015, 0.22, 8);
const toeGeo = new THREE.SphereGeometry(0.025, 8, 8);

// Create legs - initialize at visible standing positions
const legs = {};
const defaultLegPositions = {
  'FL': {hip: [0.2, 0.35, 0.12], knee: [0.2, 0.15, 0.12], toe: [0.2, 0.0, 0.12]},
  'FR': {hip: [0.2, 0.35, -0.12], knee: [0.2, 0.15, -0.12], toe: [0.2, 0.0, -0.12]},
  'BL': {hip: [-0.2, 0.35, 0.12], knee: [-0.2, 0.15, 0.12], toe: [-0.2, 0.0, 0.12]},
  'BR': {hip: [-0.2, 0.35, -0.12], knee: [-0.2, 0.15, -0.12], toe: [-0.2, 0.0, -0.12]},
};
['FL','FR','BL','BR'].forEach(name => {
  const thigh = new THREE.Mesh(segGeo, thighMat);
  const calf = new THREE.Mesh(segGeo, calfMat);
  const toe = new THREE.Mesh(toeGeo, toeMat);
  
  // Set initial positions so legs are visible at startup
  const defPos = defaultLegPositions[name];
  const hip = new THREE.Vector3(...defPos.hip);
  const knee = new THREE.Vector3(...defPos.knee);
  const toePos = new THREE.Vector3(...defPos.toe);
  
  // Position thigh at midpoint between hip and knee
  const thighMid = hip.clone().add(knee).multiplyScalar(0.5);
  thigh.position.copy(thighMid);
  thigh.rotation.set(0, 0, 0);
  thigh.lookAt(knee);
  thigh.rotateX(Math.PI/2);
  
  // Position calf at midpoint between knee and toe
  const calfMid = knee.clone().add(toePos).multiplyScalar(0.5);
  calf.position.copy(calfMid);
  calf.rotation.set(0, 0, 0);
  calf.lookAt(toePos);
  calf.rotateX(Math.PI/2);
  
  // Position toe at ground level
  toe.position.copy(toePos);
  
  scene.add(thigh, calf, toe);
  legs[name] = {thigh, calf, toe};
});

// Contact indicators
const contactGeo = new THREE.RingGeometry(0.03, 0.05, 16);
const contactMat = new THREE.MeshBasicMaterial({color: 0x00ff00, side: THREE.DoubleSide});
const contactIndicators = {};
['FL','FR','BL','BR'].forEach(name => {
  const ring = new THREE.Mesh(contactGeo, contactMat.clone());
  ring.rotation.x = -Math.PI / 2;
  ring.visible = false;
  scene.add(ring);
  contactIndicators[name] = ring;
});

// Shadow markers
const shadowGeo = new THREE.CircleGeometry(0.02, 8);
const shadowMat = new THREE.MeshBasicMaterial({color: 0xff4444, side: THREE.DoubleSide});
const shadowMarkers = [];
for(let i=0; i<4; i++){
  const m = new THREE.Mesh(shadowGeo, shadowMat);
  m.rotation.x = -Math.PI/2;
  m.position.y = 0.001;
  m.visible = false;
  scene.add(m);
  shadowMarkers.push(m);
}

// Camera control
let camTarget = new THREE.Vector3(0, 0.3, 0);
let dogX = 0, dogZ = 0;

let isDragging = false, prevMouse = {x:0,y:0};
let spherical = {theta: Math.PI/4, phi: Math.PI/3, radius: 2.5};

function updateCamera(){
  const x = camTarget.x + spherical.radius * Math.sin(spherical.phi) * Math.cos(spherical.theta);
  const y = camTarget.y + spherical.radius * Math.cos(spherical.phi);
  const z = camTarget.z + spherical.radius * Math.sin(spherical.phi) * Math.sin(spherical.theta);
  camera.position.set(x, y, z);
  camera.lookAt(camTarget);
}

renderer.domElement.addEventListener('mousedown', e => {
  isDragging = true;
  prevMouse = {x: e.clientX, y: e.clientY};
});
window.addEventListener('mouseup', () => isDragging = false);
window.addEventListener('mousemove', e => {
  if(!isDragging) return;
  const dx = e.clientX - prevMouse.x, dy = e.clientY - prevMouse.y;
  spherical.theta -= dx * 0.01;
  spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi + dy * 0.01));
  prevMouse = {x: e.clientX, y: e.clientY};
  updateCamera();
});
renderer.domElement.addEventListener('wheel', e => {
  spherical.radius = Math.max(0.5, Math.min(10, spherical.radius + e.deltaY * 0.002));
  updateCamera();
  e.preventDefault();
}, {passive: false});

window.addEventListener('keydown', e => {
  if(e.key === 'r' || e.key === 'R') recenter();
});

function recenter(){
  camTarget.set(dogX, 0.3, dogZ);
  updateCamera();
}

window.addEventListener('resize', () => {
  WIDTH = container.clientWidth;
  HEIGHT = container.clientHeight;
  camera.aspect = WIDTH / HEIGHT;
  camera.updateProjectionMatrix();
  renderer.setSize(WIDTH, HEIGHT);
});

updateCamera();

// === 2D CANVASES ===
const map = document.getElementById('map'), mapc = map.getContext('2d');
const tele = document.getElementById('tele'), telec = tele.getContext('2d');

function resizeCanvases(){
  map.width = map.clientWidth; map.height = map.clientHeight;
  tele.width = tele.clientWidth; tele.height = tele.clientHeight;
}
resizeCanvases();
window.addEventListener('resize', resizeCanvases);

// === STATE ===
let vizMode = 'hyperspace';
const GENE_NAMES = ['SIZE','DENS','LK_F','LK_M','LK_S','PL_F','PL_S','RAD','LR','W_ALV','W_DST','W_UP','W_TAU','W_CNT'];
const LABELS = GENE_NAMES.slice(0,9);

function setViz(m){
  vizMode = m;
  document.getElementById('btnHyper').classList.toggle('active', m==='hyperspace');
  document.getElementById('btnNeural').classList.toggle('active', m==='neural');
  document.getElementById('btnSigma').classList.toggle('active', m==='sigma');
  document.getElementById('vizLabel').textContent = m==='hyperspace'?'Hyperspace':(m==='neural'?'Neural Net':'Gene σ');
  fetch('/set_viz_mode',{method:'POST',body:m});
}

async function triggerDemo(){await fetch('/trigger_demo',{method:'POST'});}
async function directDemo(){await fetch('/direct_demo',{method:'POST'});}
async function stopDemo(){await fetch('/stop_demo',{method:'POST'});}

// === UPDATE 3D SCENE ===
// === UPDATE 3D SCENE ===
function updateScene(sim){
  const debugEl = document.getElementById('debug');
  
  if(!sim) {
    // No sim data at all
    return;
  }
  if(!sim.kin) {
    // sim exists but no kin - this might be normal during training
    if(debugEl && sim.height === undefined) {
      debugEl.textContent += ' | no kin data';
    }
    return;
  }
  
  const kin = sim.kin;
  const pos = kin.torso_pos;
  const rot = kin.torso_rot;
  
  dogX = pos[0]; dogZ = pos[2];
  
  torso.position.set(pos[0], pos[1], pos[2]);
  // FIXED: Use correct euler order (roll, pitch, yaw) matching Python physics
  // Python: roll=X, pitch=Y, yaw=Z | Three.js XYZ euler = (X, Y, Z)
  torso.rotation.set(rot[0], rot[1], rot[2]);
  
  ['FL','FR','BL','BR'].forEach((name, idx) => {
    const leg = kin.legs[name];
    const parts = legs[name];
    
    const hip = new THREE.Vector3(leg.hip[0], leg.hip[1], leg.hip[2]);
    const knee = new THREE.Vector3(leg.knee[0], leg.knee[1], leg.knee[2]);
    const toe = new THREE.Vector3(leg.toe[0], leg.toe[1], leg.toe[2]);
    
    // Thigh segment
    const thighMid = hip.clone().add(knee).multiplyScalar(0.5);
    parts.thigh.position.copy(thighMid);
    // Reset rotation before lookAt to prevent accumulation
    parts.thigh.rotation.set(0, 0, 0);
    parts.thigh.lookAt(knee);
    parts.thigh.rotateX(Math.PI/2);
    
    // Calf segment
    const calfMid = knee.clone().add(toe).multiplyScalar(0.5);
    parts.calf.position.copy(calfMid);
    // Reset rotation before lookAt to prevent accumulation
    parts.calf.rotation.set(0, 0, 0);
    parts.calf.lookAt(toe);
    parts.calf.rotateX(Math.PI/2);
    
    // Toe sphere
    parts.toe.position.copy(toe);
    
    const ci = contactIndicators[name];
    if(sim.contacts && sim.contacts[idx] > 0.5){
      ci.visible = true;
      ci.position.set(toe.x, 0.002, toe.z);
      ci.material.color.setHex(0x00ff00);
    } else {
      ci.visible = false;
    }
  });
  
  const shadows = sim.shadows || {};
  let si = 0;
  for(const key in shadows){
    if(si >= shadowMarkers.length) break;
    const s = shadows[key];
    shadowMarkers[si].visible = true;
    shadowMarkers[si].position.set(s[0], 0.002, s[1]);
    si++;
  }
  for(; si < shadowMarkers.length; si++){
    shadowMarkers[si].visible = false;
  }
  
  gridHelper.position.x = Math.floor(dogX / 2) * 2;
  gridHelper.position.z = Math.floor(dogZ / 2) * 2;
  ground.position.x = gridHelper.position.x;
  ground.position.z = gridHelper.position.z;
  
  camTarget.x += (dogX - camTarget.x) * 0.05;
  camTarget.z += (dogZ - camTarget.z) * 0.05;
  updateCamera();
}

// === HYPERSPACE VIZ ===
let axisX=0, axisY=7, lastSwitch=Date.now();
function drawHyper(pop, hist){
  const W=map.width, H=map.height;
  mapc.fillStyle='#0f1720'; mapc.fillRect(0,0,W,H);
  
  mapc.strokeStyle='#223044'; mapc.lineWidth=1;
  mapc.beginPath(); mapc.moveTo(0,H/2); mapc.lineTo(W,H/2); mapc.stroke();
  mapc.beginPath(); mapc.moveTo(W/2,0); mapc.lineTo(W/2,H); mapc.stroke();
  
  mapc.fillStyle='#778'; mapc.font='10px monospace';
  mapc.fillText(`X:${LABELS[axisX]}`,W-70,H-8);
  mapc.fillText(`Y:${LABELS[axisY]}`,8,14);
  
  const proj = v => [30+v[axisX]*(W-60), (H-30)-v[axisY]*(H-60)];
  
  if(hist && hist.length>1){
    mapc.strokeStyle='rgba(34,211,238,0.5)'; mapc.lineWidth=2;
    mapc.beginPath();
    hist.forEach((v,i) => {const[x,y]=proj(v); i===0?mapc.moveTo(x,y):mapc.lineTo(x,y);});
    mapc.stroke();
  }
  
  if(pop) pop.forEach((v,i) => {
    const elite = v[9]>0.5;
    const[x,y] = proj(v);
    mapc.beginPath(); mapc.arc(x,y,elite?5:2,0,Math.PI*2);
    mapc.fillStyle = elite?'#ff5500':'#44556c';
    mapc.fill();
  });
  
  if(Date.now()-lastSwitch>3000){
    lastSwitch=Date.now();
    axisX=Math.floor(Math.random()*9);
    do{axisY=Math.floor(Math.random()*9);}while(axisY===axisX);
  }
}

// === NEURAL VIZ ===
function drawNeural(brain){
  const W=map.width, H=map.height;
  mapc.fillStyle='#0f1720'; mapc.fillRect(0,0,W,H);
  
  if(!brain || !brain.activations){
    mapc.fillStyle='#556'; mapc.font='12px sans-serif';
    mapc.fillText('Waiting for demo…',W/2-50,H/2);
    return;
  }
  
  const acts=brain.activations, nViz=acts.length;
  const cx=W/2, cy=H/2, rad=Math.min(W,H)*0.38;
  const nPos = i => {const th=(i/nViz)*2*Math.PI-Math.PI/2; return [cx+rad*Math.cos(th), cy+rad*Math.sin(th)];};
  
  const poolInfo = brain.pool_info || {};
  const pb = poolInfo.pool_boundaries || [0, nViz/3, 2*nViz/3, nViz];
  const getPoolColor = idx => {
    if(idx < pb[1]/brain.n_reservoir*nViz) return '#ef4444';
    if(idx < pb[2]/brain.n_reservoir*nViz) return '#f59e0b';
    return '#3b82f6';
  };
  
  const links = brain.links || [], lw = brain.link_weights || [];
  mapc.lineWidth = 0.5;
  for(let i=0; i<Math.min(links.length,200); i++){
    const[src,dst] = links[i];
    const si = Math.floor(src/brain.n_reservoir*nViz);
    const di = Math.floor(dst/brain.n_reservoir*nViz);
    if(si>=nViz || di>=nViz) continue;
    const[x1,y1]=nPos(si), [x2,y2]=nPos(di);
    const w = lw[i]||0;
    const alpha = Math.min(0.6, Math.abs(acts[si]||0)*Math.abs(w)*2+0.05);
    mapc.strokeStyle = w>0 ? `rgba(34,211,238,${alpha})` : `rgba(255,100,100,${alpha})`;
    mapc.beginPath(); mapc.moveTo(x1,y1); mapc.lineTo(x2,y2); mapc.stroke();
  }
  
  for(let i=0; i<nViz; i++){
    const[x,y] = nPos(i);
    const act = acts[i]||0;
    const r = 2 + Math.abs(act)*4;
    mapc.fillStyle = getPoolColor(i);
    mapc.globalAlpha = 0.3 + Math.abs(act)*0.7;
    mapc.beginPath(); mapc.arc(x,y,r,0,Math.PI*2); mapc.fill();
  }
  mapc.globalAlpha = 1;
}

// === SIGMA VIZ ===
function drawSigma(sigma, sensitivity){
  const W=map.width, H=map.height;
  mapc.fillStyle='#0f1720'; mapc.fillRect(0,0,W,H);
  
  if(!sigma || sigma.length===0){
    mapc.fillStyle='#556'; mapc.font='12px sans-serif';
    mapc.fillText('No sigma data…',W/2-40,H/2);
    return;
  }
  
  const n=sigma.length, barW=(W-60)/n-2;
  const maxSig=0.2;
  
  mapc.fillStyle='#778'; mapc.font='9px monospace';
  mapc.fillText('σ (cyan) | sens (red)',10,12);
  
  for(let i=0; i<n; i++){
    const x=30+i*(barW+2);
    const sigH = Math.min(1,sigma[i]/maxSig)*(H-50);
    const sensH = sensitivity ? Math.min(1,sensitivity[i])*(H-50) : 0;
    
    mapc.fillStyle='rgba(34,211,238,0.7)';
    mapc.fillRect(x, H-25-sigH, barW/2-1, sigH);
    
    mapc.fillStyle='rgba(239,68,68,0.5)';
    mapc.fillRect(x+barW/2, H-25-sensH, barW/2-1, sensH);
    
    mapc.fillStyle='#667';
    mapc.save();
    mapc.translate(x+barW/2, H-10);
    mapc.rotate(-Math.PI/3);
    mapc.fillText(GENE_NAMES[i],0,0);
    mapc.restore();
  }
}

function drawMap(pop, hist, brain, sigma, sensitivity){
  if(vizMode==='sigma') drawSigma(sigma, sensitivity);
  else if(vizMode==='neural') drawNeural(brain);
  else drawHyper(pop, hist);
}

// === TELEMETRY ===
let teleHist = [];
function drawTele(sim){
  const W=tele.width, H=tele.height;
  telec.fillStyle='#0f1720'; telec.fillRect(0,0,W,H);
  
  if(!sim || sim.t===undefined) return;
  
  teleHist.push([sim.t, sim.r||0, sim.distance||0, sim.height||0, sim.vel||0]);
  if(teleHist.length>200) teleHist.shift();
  
  const t0=teleHist[0][0], t1=teleHist[teleHist.length-1][0], dt=Math.max(0.1,t1-t0);
  const xOf = t => 30+(t-t0)/dt*(W-50);
  const yOf = (val,vmin,vmax) => (H-20)-(val-vmin)/(vmax-vmin+0.01)*(H-35);
  
  const rV=teleHist.map(x=>x[1]), rmin=Math.min(...rV,-3), rmax=Math.max(...rV,3);
  telec.strokeStyle='rgba(255,157,0,0.8)'; telec.lineWidth=1.5; telec.beginPath();
  teleHist.forEach((v,i) => {const x=xOf(v[0]),y=yOf(v[1],rmin,rmax); i===0?telec.moveTo(x,y):telec.lineTo(x,y);});
  telec.stroke();
  
  const dV=teleHist.map(x=>x[2]), dmin=Math.min(...dV,0), dmax=Math.max(...dV,1);
  telec.strokeStyle='rgba(34,211,238,0.8)'; telec.beginPath();
  teleHist.forEach((v,i) => {const x=xOf(v[0]),y=yOf(v[2],dmin,dmax); i===0?telec.moveTo(x,y):telec.lineTo(x,y);});
  telec.stroke();
  
  const hV=teleHist.map(x=>x[3]), hmin=0, hmax=Math.max(...hV,0.5);
  telec.strokeStyle='rgba(147,197,253,0.6)'; telec.beginPath();
  teleHist.forEach((v,i) => {const x=xOf(v[0]),y=yOf(v[3],hmin,hmax); i===0?telec.moveTo(x,y):telec.lineTo(x,y);});
  telec.stroke();
  
  telec.fillStyle='#ff9d00'; telec.fillRect(W-90,8,12,3);
  telec.fillStyle='#778'; telec.font='9px sans-serif'; telec.fillText('reward',W-75,12);
  telec.fillStyle='#22d3ee'; telec.fillRect(W-90,18,12,3);
  telec.fillText('dist',W-75,22);
  telec.fillStyle='#93c5fd'; telec.fillRect(W-90,28,12,3);
  telec.fillText('height',W-75,32);
}

// === POLLING ===
let pollCount = 0;
async function poll(){
  pollCount++;
  const debugEl = document.getElementById('debug');
  
  try {
    const r = await fetch('/status');
    if(!r.ok) {
      debugEl.textContent = `Poll ${pollCount}: HTTP ${r.status}`;
      return;
    }
    const d = await r.json();
    
    // Debug info
    const hasKin = d.sim && d.sim.kin;
    const simKeys = d.sim ? Object.keys(d.sim).join(',') : 'none';
    debugEl.textContent = `Poll ${pollCount}: mode=${d.mode}, hasKin=${hasKin}, simKeys=${simKeys.slice(0,50)}`;
    
    document.getElementById('status').textContent = d.status || '—';
    document.getElementById('mode').textContent = d.mode || '—';
    document.getElementById('gen').textContent = d.gen || 0;
    document.getElementById('best').textContent = (d.best_distance||0).toFixed(3);
    document.getElementById('bestSteps').textContent = (d.best_steps||0).toFixed(1);
    document.getElementById('stag').textContent = d.stagnation_count||0;
    document.getElementById('cid').textContent = d.id || '—';
    
    const sim = d.sim || {};
    document.getElementById('hval').textContent = (sim.height||0).toFixed(3);
    document.getElementById('rpval').textContent = `${(sim.roll||0).toFixed(2)}/${(sim.pitch||0).toFixed(2)}`;
    document.getElementById('dval').textContent = (sim.distance||0).toFixed(3);
    document.getElementById('vval').textContent = (sim.vel||0).toFixed(3);
    document.getElementById('cval').textContent = (sim.contacts||[0,0,0,0]).map(c=>c>0.5?'●':'○').join(' ');
    document.getElementById('sval').textContent = sim.steps||'—';
    
    // CPG phase display
    const cpgPhase = sim.cpg_phase || [];
    if(cpgPhase.length >= 8) {
      const legs = ['FL','FR','BL','BR'];
      let phaseStr = '';
      for(let i=0; i<4; i++) {
        const sin = cpgPhase[i*2], cos = cpgPhase[i*2+1];
        const phase = Math.atan2(sin, cos) / (2*Math.PI) + 0.5;
        const isSwing = phase > 0.6;
        phaseStr += `${legs[i]}:${isSwing?'↑':'↓'} `;
      }
      document.getElementById('cpgPhase').textContent = phaseStr;
    }
    
    const hp = d.params || {};
    document.getElementById('hp').textContent = 
`reservoir = ${hp.n_reservoir||'—'}
density   = ${hp.density?.toFixed(3)||'—'}
leak_fast = ${hp.leak_fast?.toFixed(3)||'—'}
leak_med  = ${hp.leak_med?.toFixed(3)||'—'}
leak_slow = ${hp.leak_slow?.toFixed(3)||'—'}
spect_rad = ${hp.spectral_radius?.toFixed(3)||'—'}
lr        = ${hp.lr?.toExponential(2)||'—'}`;
    
    const rw = hp.reward_w || [0,0,0,0,0];
    document.getElementById('rw').textContent = 
`alive=${rw[0]?.toFixed(2)} dist=${rw[1]?.toFixed(2)} up=${rw[2]?.toFixed(2)} tau=${rw[3]?.toFixed(2)} cnt=${rw[4]?.toFixed(2)}`;
    
    const pf=(hp.pool_fast*100||0).toFixed(0), pm=(hp.pool_med*100||0).toFixed(0), ps=(hp.pool_slow*100||0).toFixed(0);
    document.getElementById('pools').innerHTML = `<span class="tag tag-fast">${pf}%</span><span class="tag tag-med">${pm}%</span><span class="tag tag-slow">${ps}%</span>`;
    
    document.getElementById('logs').textContent = (d.logs||[]).slice(0,12).join('\n');
    
    updateScene(sim);
    drawMap(d.pop_vectors||[], d.history_vectors||[], d.brain||{}, d.gene_sigma||[], d.gene_sensitivity||[]);
    drawTele(sim);
    
  } catch(e){ 
    console.error('Poll error:', e);
    const debugEl = document.getElementById('debug');
    if(debugEl) debugEl.textContent = 'ERROR: ' + e.message;
  }
}

function animate(){
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
}
animate();

setInterval(poll, 150);  // Slower polling - 150ms instead of 80ms
poll();
console.log('Dog Lab v7 UI initialized, polling started at 150ms interval');
</script>
</body>
</html>
"""

app = Flask(__name__)


@app.get("/")
def index() -> Response:
    return Response(HTML, mimetype="text/html")


@app.get("/test")
def test_endpoint():
    """Simple test endpoint to verify API works."""
    return jsonify({
        "ok": True,
        "time": time.time(),
        "message": "API is working"
    })


@app.get("/status")
def status():
    try:
        # Get a snapshot of state - use short lock
        with _STATE_LOCK:
            state_copy = {
                "status": SYSTEM_STATE.get("status", "UNKNOWN"),
                "generation": SYSTEM_STATE.get("generation", 0),
                "mode": SYSTEM_STATE.get("mode", "UNKNOWN"),
                "current_id": SYSTEM_STATE.get("current_id", "—"),
                "logs": list(SYSTEM_STATE.get("logs", [])[:12]),  # Copy to avoid issues
                "hyperparams": dict(SYSTEM_STATE.get("hyperparams", {})),
                "pop_vectors": SYSTEM_STATE.get("pop_vectors", []),
                "history_vectors": SYSTEM_STATE.get("history_vectors", []),
                "sim_view": SYSTEM_STATE.get("sim_view", {}),
                "brain_view": SYSTEM_STATE.get("brain_view", {}),
                "demo_resets": SYSTEM_STATE.get("demo_resets", 0),
                "gene_sigma": SYSTEM_STATE.get("gene_sigma", []),
                "gene_sensitivity": SYSTEM_STATE.get("gene_sensitivity", []),
                "stagnation_count": SYSTEM_STATE.get("stagnation_count", 0),
                "best_distance": SYSTEM_STATE.get("best_distance", -1e9),
                "best_steps": SYSTEM_STATE.get("best_steps", 0.0),
                "kp": SYSTEM_STATE.get("kp", DEFAULT_KP),
                "kd": SYSTEM_STATE.get("kd", DEFAULT_KD),
            }
        
        # Build result outside of lock
        result = {
            "status": state_copy["status"],
            "gen": state_copy["generation"],
            "mode": state_copy["mode"],
            "id": state_copy["current_id"],
            "logs": state_copy["logs"],
            "params": state_copy["hyperparams"],
            "pop_vectors": state_copy["pop_vectors"],
            "history_vectors": state_copy["history_vectors"],
            "sim": state_copy["sim_view"],
            "brain": state_copy["brain_view"],
            "demo_resets": state_copy["demo_resets"],
            "gene_sigma": state_copy["gene_sigma"],
            "gene_sensitivity": state_copy["gene_sensitivity"],
            "stagnation_count": state_copy["stagnation_count"],
            "best_distance": state_copy["best_distance"],
            "best_steps": state_copy["best_steps"],
            "kp": state_copy["kp"],
            "kd": state_copy["kd"],
        }
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()[:500],
            "status": "ERROR",
            "mode": "ERROR",
            "gen": 0,
            "id": "ERROR",
            "logs": [f"Status endpoint error: {str(e)}"],
            "params": {},
            "sim": {},
            "brain": {},
            "pop_vectors": [],
            "history_vectors": [],
        })


@app.post("/trigger_demo")
def trigger_demo():
    add_log("📡 Demo button clicked!")
    
    with _STATE_LOCK:
        current_mode = SYSTEM_STATE["mode"]
        has_genome = SYSTEM_STATE.get("best_genome") is not None
        
        add_log(f"📡 Mode={current_mode}, has_genome={has_genome}")
        
        if current_mode == "DEMO":
            add_log("⚠️ Already in demo mode")
            return jsonify({"error": "Already running demo"}), 400
        
        # If no best genome yet, create a default one for testing
        if not SYSTEM_STATE["best_genome"]:
            add_log("⚠️ No champion yet, creating default genome for demo")
            SYSTEM_STATE["best_genome"] = [0.5] * GENOME_DIM  # Middle-of-range defaults
        
        SYSTEM_STATE["manual_demo_request"] = True
        add_log("✅ Demo request flag set!")
    
    return jsonify({"status": "ok"})


@app.post("/direct_demo")
def direct_demo():
    """Direct demo that runs immediately in a separate thread."""
    add_log("🎯 Direct demo requested!")
    
    with _STATE_LOCK:
        if SYSTEM_STATE["mode"] == "DEMO":
            return jsonify({"error": "Already in demo"}), 400
        
        # Create default genome if needed
        if not SYSTEM_STATE["best_genome"]:
            SYSTEM_STATE["best_genome"] = [0.5] * GENOME_DIM
        
        genome = SYSTEM_STATE["best_genome"]
        weights = SYSTEM_STATE.get("best_weights")
        kp = SYSTEM_STATE.get("kp", DEFAULT_KP)
        kd = SYSTEM_STATE.get("kd", DEFAULT_KD)
    
    def run_direct_demo():
        add_log("🎬 Direct demo thread started")
        try:
            with _STATE_LOCK:
                SYSTEM_STATE["mode"] = "DEMO"
                SYSTEM_STATE["demo_stop"] = False
                SYSTEM_STATE["status"] = "👁️ DIRECT DEMO"
            
            params = GeneDecoder.decode(np.array(genome, dtype=np.float64))
            env = DogEnv3D(kp=kp, kd=kd, reward_w=np.asarray(params["reward_w"], dtype=np.float64))
            agent = LiquidAgent(params).to(DEVICE)
            
            # Try to load weights
            if weights:
                try:
                    agent.load_state_dict(weights, strict=True)
                    add_log("✅ Loaded weights")
                except:
                    add_log("⚠️ Using random weights")
            else:
                add_log("⚠️ No weights, using random")
            
            agent.eval()
            obs = env.reset().to(DEVICE)
            prev_action = torch.zeros(1, ACTION_DIM, dtype=torch.float32).to(DEVICE)
            h = agent.init_hidden(1).to(DEVICE)
            
            add_log("🎮 Demo loop running...")
            steps = 0
            resets = 0
            
            while True:
                with _STATE_LOCK:
                    if SYSTEM_STATE["demo_stop"] or SYSTEM_STATE["mode"] != "DEMO":
                        break
                
                a_np, h, mu_np, v_est = agent.act_deterministic(obs, prev_action, h)
                obs2, r, done, info = env.step(a_np)
                obs = obs2.to(DEVICE)
                prev_action = torch.from_numpy(a_np).float().unsqueeze(0).to(DEVICE)
                steps += 1
                
                # Update visualization
                kin = env.model.get_all_kinematics()
                cpg_targets = env.cpg.get_target_angles()
                cpg_phase = env.cpg.get_phase_encoding()
                
                with _STATE_LOCK:
                    SYSTEM_STATE["sim_view"] = {
                        "t": float(env.model.t),
                        "q": env.model.q.tolist(),
                        "qd": env.model.qd.tolist(),
                        "r": float(r),
                        "distance": float(info["distance"]),
                        "vel": float(info["vel"]),
                        "height": float(info["height"]),
                        "roll": float(info["roll"]),
                        "pitch": float(info["pitch"]),
                        "yaw": float(info["yaw"]),
                        "lateral": float(info["lateral"]),
                        "kin": kin,
                        "shadows": {k: v.tolist() for k, v in env.model.floor.shadows.items()},
                        "act": [float(x) for x in a_np.tolist()],
                        "steps": steps,
                        "contacts": env.contacts,
                        "cpg_targets": cpg_targets.tolist(),
                        "cpg_phase": cpg_phase.tolist(),
                    }
                
                if done:
                    resets += 1
                    if resets % 5 == 0:
                        add_log(f"👁️ Reset x{resets} dist={info['distance']:.2f}")
                    obs = env.reset().to(DEVICE)
                    prev_action = torch.zeros(1, ACTION_DIM, dtype=torch.float32).to(DEVICE)
                    h = agent.init_hidden(1).to(DEVICE)
                    steps = 0
                
                time.sleep(0.02)  # ~50 FPS
            
        except Exception as e:
            import traceback
            add_log(f"❌ Demo error: {str(e)[:60]}")
            add_log(traceback.format_exc()[:200])
        finally:
            with _STATE_LOCK:
                SYSTEM_STATE["mode"] = "TRAINING"
                SYSTEM_STATE["demo_stop"] = False
            add_log("🎬 Direct demo ended")
    
    # Start demo in separate thread
    demo_thread = threading.Thread(target=run_direct_demo, daemon=True)
    demo_thread.start()
    
    return jsonify({"status": "ok", "message": "Direct demo started"})


@app.post("/stop_demo")
def stop_demo():
    with _STATE_LOCK:
        SYSTEM_STATE["demo_stop"] = True
        SYSTEM_STATE["mode"] = "TRAINING"
    add_log("⏹ Demo stopped")
    return jsonify({"status": "ok"})


@app.post("/set_viz_mode")
def set_viz_mode():
    mode = request.get_data(as_text=True) or "hyperspace"
    with _STATE_LOCK:
        SYSTEM_STATE["viz_mode"] = mode
    return jsonify({"status": "ok"})


def main() -> None:
    torch.set_num_threads(1)
    add_log("🦿 DOG LAB v7 3D — IRON WOLF (CPG Enhanced)")
    add_log(f"🧪 Pop={POP_SIZE} Workers={CORE_COUNT} Elites={N_ELITES}")
    add_log(f"📐 3D Model: {TOTAL_DOF} DOF (6 torso + 12 joints)")
    add_log(f"📊 Obs={OBS_DIM} Act={ACTION_DIM}")
    add_log(f"🎛️ CPG: freq={CPG_FREQUENCY}Hz residual={RESIDUAL_SCALE}")
    add_log("🎮 Controls: drag=rotate, scroll=zoom, R=recenter")
    
    engine = EvolutionEngine()
    engine.start()
    
    with _STATE_LOCK:
        SYSTEM_STATE["status"] = "READY"
    
    port = int(os.environ.get("PORT", "5005"))
    host = os.environ.get("HOST", "127.0.0.1")
    url = f"http://{host}:{port}"
    add_log(f"🌐 {url}")
    
    try:
        time.sleep(0.8)
        webbrowser.open(url)
    except Exception:
        pass
    
    app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except Exception:
        pass
    main()
