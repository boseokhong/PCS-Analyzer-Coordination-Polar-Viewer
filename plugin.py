"""
Coordination Polar Viewer  (v0.1.2)
=====================================

Standalone / plug-in viewer for comparing multiple XYZ coordination
structures in a metal-centred 2D polar coordinate representation.
Part of the PCS Analyzer suite.

Improvements over v0.1.2
-------------------------
F – Session management
  F1  Save and restore full viewer sessions (.cpvsession / JSON)
  F2  Preserve loaded XYZ paths, reference selection, per-structure colours,
      memo text, metal centres, manual metal coordinates, visibility,
      axis-alignment state, coordination-sphere selections, and display options

A – Analysis accuracy
  A1  Kabsch-aligned RMSD  (optimal rotation before distance calculation)
  A2  Automatic metal detection from element list → dropdown selection
  A3  Metal-ligand distance table + ligand-metal-ligand angle table in
      a dedicated "Analysis" tab

B – Visualisation quality
  B1  Displacement arrows  (reference → comparison atom movement on polar plot)
  B2  Dynamic colour palette  (matplotlib colormap, unlimited structures)
      + per-structure colour picker button

C – Interaction
  C1  Atom hover tooltip  (element, 1-based index, distance from metal)
  C2  Plot right-click context menu  (save PNG/SVG/PDF, copy atom info,
      toggle displacement arrows)
  C3  Per-structure memo/annotation field  (shown in legend)

D – UI / style
  D1  Polished ttk.Style  (science-tool aesthetic, consistent spacing)
  D2  PanedWindow layout  (left/right pane draggable)
  D3  Status bar  (structure count, last action)
  D4  Export: PNG / SVG / PDF / CSV (numeric data)

E – System
  E1  config.json  (auto-save & restore last session settings)

Usage
-----
Standalone:
    python xyz_compare_viewer.py

From host Tkinter app:
    from xyz_compare_viewer import open_xyz_compare_viewer
    open_xyz_compare_viewer(parent=root)
"""

from __future__ import annotations

import os
import json
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import numpy as np

import matplotlib
try:
    if "tk" not in matplotlib.get_backend().lower():
        matplotlib.use("TkAgg")
except Exception:
    pass

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.lines import Line2D

try:
    import mplcursors
    _MPLCURSORS = True
except ImportError:
    _MPLCURSORS = False

PLUGIN_INFO = {
    "id": "coordination_polar_viewer",
    "name": "Coordination Polar Viewer",
    "version": "0.1.2",
    "author": "Boseok Hong",
    "description": (
        "Compare coordination structures of metal complexes in a metal-centred polar coordinate view."
    ),
    "type": "window",
    "standalone": True,
    "dependencies": ["numpy", "matplotlib"],
}

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "coordination_polar_viewer_config.json")
SESSION_FILETYPES = [("Coordination Polar Viewer session", "*.cpvsession"), ("JSON files", "*.json"), ("All files", "*.*")]

# ─────────────────────────────────────────────────────────────────────────────
# Chemistry constants
# ─────────────────────────────────────────────────────────────────────────────

FALLBACK_CPK_COLORS: Dict[str, str] = {
    "H": "#ffffff", "C": "#444444", "N": "#3050f8", "O": "#ff0d0d",
    "F": "#90e050", "P": "#ff8000", "S": "#ffff30", "Cl": "#1ff01f",
    "Br": "#a62929", "I": "#940094", "Pt": "#d0d0d0", "Pd": "#b0b0b0",
    "Ru": "#248f8f", "Rh": "#0a7d8c", "Ir": "#175487", "Au": "#ffd123",
    "Fe": "#e06633", "Co": "#f090a0", "Ni": "#50d050", "Cu": "#c88033",
    "Zn": "#7d80b0", "Np": "#4a90e2", "U": "#4a90e2", "Th": "#4a90e2",
    "default": "#aaaaaa",
}

FALLBACK_COVALENT_RADII: Dict[str, float] = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "P": 1.07, "S": 1.05, "Cl": 1.02, "Br": 1.20, "I": 1.39,
    "Pt": 1.23, "Pd": 1.20, "Ru": 1.25, "Rh": 1.25, "Ir": 1.32,
    "Au": 1.23, "Fe": 1.25, "Co": 1.26, "Ni": 1.21, "Cu": 1.32,
    "Zn": 1.22, "Np": 1.90, "U": 1.96, "Th": 2.06, "default": 0.80,
}

# Transition metals for auto-detection (A2)
TRANSITION_METALS = {
    "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
    "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
    "La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy",
    "Ho","Er","Tm","Yb","Lu",
    "Ac","Th","Pa","U","Np","Pu","Am","Cm",
}

try:
    from logic.chem_constants import CPK_COLORS, covalent_radii
except Exception:
    CPK_COLORS = FALLBACK_CPK_COLORS
    covalent_radii = FALLBACK_COVALENT_RADII

# ─────────────────────────────────────────────────────────────────────────────
# Shannon ionic radii  (6-coordinate, most common oxidation state, Å)
# Source: Shannon, Acta Cryst. A32, 751 (1976)
# ─────────────────────────────────────────────────────────────────────────────
SHANNON_RADII: Dict[str, float] = {
    "Li": 0.76, "Na": 1.02, "K":  1.38, "Rb": 1.52, "Cs": 1.67,
    "Be": 0.45, "Mg": 0.72, "Ca": 1.00, "Sr": 1.18, "Ba": 1.35,
    "Sc": 0.745,"Ti": 0.670,"V":  0.640,"Cr": 0.615,"Mn": 0.830,
    "Fe": 0.645,"Co": 0.610,"Ni": 0.690,"Cu": 0.730,"Zn": 0.740,
    "Y":  0.900,"Zr": 0.720,"Nb": 0.640,"Mo": 0.650,"Tc": 0.645,
    "Ru": 0.680,"Rh": 0.665,"Pd": 0.860,"Ag": 1.150,"Cd": 0.950,
    "La": 1.032,"Ce": 1.010,"Pr": 0.990,"Nd": 0.983,"Sm": 0.958,
    "Eu": 0.947,"Gd": 0.938,"Tb": 0.923,"Dy": 0.912,"Ho": 0.901,
    "Er": 0.890,"Tm": 0.880,"Yb": 0.868,"Lu": 0.861,
    "Hf": 0.710,"Ta": 0.640,"W":  0.600,"Re": 0.630,"Os": 0.630,
    "Ir": 0.625,"Pt": 0.625,"Au": 1.370,"Hg": 1.190,
    "Al": 0.535,"Ga": 0.620,"In": 0.800,"Tl": 1.500,
    "Ge": 0.530,"Sn": 0.690,"Pb": 1.190,
    "As": 0.580,"Sb": 0.760,"Bi": 1.030,
    "Th": 0.940,"U":  0.730,"Np": 0.720,"Pu": 0.710,
}

# ─────────────────────────────────────────────────────────────────────────────
# Geometry analysis helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_bonded(elements: List[str], coords: np.ndarray,
                metal_idx: int, metal_pos: np.ndarray,
                scale: float = 1.25) -> List[int]:
    """Return indices of atoms bonded to metal (1st coordination sphere)."""
    m_el = elements[metal_idx]
    rm   = covalent_radii.get(m_el, covalent_radii["default"])
    bonded = []
    for i, (el, coord) in enumerate(zip(elements, coords)):
        if i == metal_idx:
            continue
        rl = covalent_radii.get(el, covalent_radii["default"])
        if np.linalg.norm(coord - metal_pos) < scale * (rm + rl):
            bonded.append(i)
    return bonded


def _lml_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """L-M-L angle in degrees."""
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(
        np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))))


def calc_tau4(angles_deg: List[float]) -> Optional[float]:
    """
    τ₄ index for 4-coordinate geometry (Yang et al., Dalton Trans. 2007).
    τ₄ = 1.0 → tetrahedral,  0.0 → square planar.
    Requires exactly 6 L-M-L angles (all pairs from 4 ligands).
    """
    if len(angles_deg) < 6:
        return None
    srt = sorted(angles_deg, reverse=True)
    alpha, beta = srt[0], srt[1]
    return float((360.0 - alpha - beta) / 141.0)


def calc_tau5(angles_deg: List[float]) -> Optional[float]:
    """
    τ₅ index for 5-coordinate geometry (Addison & Reedijk, J. Chem. Soc. 1984).
    τ₅ = 1.0 → trigonal bipyramidal,  0.0 → square pyramidal.
    Requires at least 10 L-M-L angles (all pairs from 5 ligands).
    """
    if len(angles_deg) < 10:
        return None
    srt = sorted(angles_deg, reverse=True)
    alpha, beta = srt[0], srt[1]
    return float((alpha - beta) / 60.0)


def calc_sigma_trans(elements: List[str], coords: np.ndarray,
                     metal_idx: int, metal_pos: np.ndarray,
                     bonded: List[int]) -> List[Tuple[int, int, float, float]]:
    """
    For each trans pair (L-M-L angle ≥ 150°), return
    (i, j, angle_deg, |d_i - d_j|) as trans-influence indicator.
    """
    results = []
    vecs = {i: coords[i] - metal_pos for i in bonded}
    dists = {i: float(np.linalg.norm(v)) for i, v in vecs.items()}
    for a in range(len(bonded)):
        for b in range(a + 1, len(bonded)):
            i, j = bonded[a], bonded[b]
            angle = _lml_angle(vecs[i], vecs[j])
            if angle >= 150.0:
                results.append((i, j, angle, abs(dists[i] - dists[j])))
    return results


def calc_angular_deviation(angles_deg: List[float],
                            cn: int) -> Optional[float]:
    """
    Σ = sum of |observed - ideal| for all L-M-L angles.
    Ideal angles depend on coordination number and assumed geometry.
    """
    ideal_map = {
        2:  [180.0],
        3:  [120.0, 120.0, 120.0],
        4:  [109.5] * 6,                  # tetrahedral default
        5:  [120.0]*3 + [90.0]*6 + [180.0],  # tbp
        6:  [90.0]*12 + [180.0]*3,        # octahedral
        7:  [72.0]*5 + [90.0]*5 + [144.0]*5,
        8:  [70.5]*12 + [109.5]*4,
    }
    ideals = ideal_map.get(cn)
    if ideals is None or len(angles_deg) != len(ideals):
        return None
    srt_obs  = sorted(angles_deg)
    srt_ideal = sorted(ideals)
    return float(sum(abs(o - i) for o, i in zip(srt_obs, srt_ideal)))


def polar_stats(elements: List[str], coords: np.ndarray,
                metal_pos: np.ndarray, bonded: List[int],
                proj: str = "0-180"
                ) -> Dict[str, float]:
    """
    Compute θ and r statistics for the coordination sphere atoms
    in the polar plot coordinate system.
    """
    if not bonded:
        return {}
    rel   = coords[bonded] - metal_pos
    r     = np.linalg.norm(rel, axis=1)
    norms = r + 1e-15
    theta = np.arccos(np.clip(rel[:, 2] / norms, -1.0, 1.0))
    if proj == "0-90 folded":
        theta = np.minimum(theta, np.pi - theta)
    theta_deg = np.degrees(theta)
    return {
        "r_mean":   float(np.mean(r)),
        "r_std":    float(np.std(r)),
        "r_min":    float(np.min(r)),
        "r_max":    float(np.max(r)),
        "theta_mean": float(np.mean(theta_deg)),
        "theta_std":  float(np.std(theta_deg)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class XYZStructure:
    label: str
    path: str
    elements: List[str]
    coords: np.ndarray
    color: str = "#1a8c5e"
    memo: str = ""
    visible: bool = True         # shown in the current session
    metal_index: int = 0        # 0-based atom index used as metal centre
    use_manual: bool = False    # if True, use manual XYZ instead of atom index
    mx: float = 0.0
    my: float = 0.0
    mz: float = 0.0
    # ── Axis alignment ──
    use_align: bool = False            # apply rotation to align axis → z
    align_axis: "np.ndarray | None" = None   # detected/manual rotation axis (3,)
    align_Cn: int = 0                  # detected symmetry order (0 = SVD fallback)
    coord_sphere: "List[int] | None" = None  # 0-based indices of coord sphere atoms


def load_xyz(path: str) -> Tuple[List[str], np.ndarray]:
    """Load a simple XYZ file (first frame only)."""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        raise ValueError("Empty XYZ file.")

    try:
        n_atoms = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError("First line of XYZ must contain the atom count.") from exc

    elements, coords = [], []
    for line in lines[2: 2 + n_atoms]:
        parts = line.split()
        if len(parts) < 4:
            continue
        elements.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

    if len(elements) != n_atoms:
        raise ValueError(
            f"XYZ atom count mismatch: expected {n_atoms}, got {len(elements)}."
        )
    return elements, np.asarray(coords, dtype=float)


def guess_bonds(elements: List[str], coords: np.ndarray, scale: float = 1.25):
    bonds = []
    n = len(elements)
    for i in range(n):
        ri = covalent_radii.get(elements[i], covalent_radii.get("default", 0.80))
        for j in range(i + 1, n):
            rj = covalent_radii.get(elements[j], covalent_radii.get("default", 0.80))
            if np.linalg.norm(coords[i] - coords[j]) < scale * (ri + rj):
                bonds.append((i, j))
    return bonds


def cpk_color(element: str) -> str:
    return CPK_COLORS.get(element, CPK_COLORS.get("default", "#aaaaaa"))


# ─────────────────────────────────────────────────────────────────────────────
# A1  Kabsch algorithm
# ─────────────────────────────────────────────────────────────────────────────

def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Return (rmsd, P_rotated) where P_rotated is P optimally rotated onto Q.
    Both P and Q are (N,3) arrays.  Centroids are removed internally.
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    p_mean = P.mean(axis=0)
    q_mean = Q.mean(axis=0)
    Pc = P - p_mean
    Qc = Q - q_mean
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    P_rot = Pc @ R.T + q_mean
    rmsd = float(np.sqrt(np.mean(np.sum((P_rot - Q) ** 2, axis=1))))
    return rmsd, P_rot


# ─────────────────────────────────────────────────────────────────────────────
# Axis alignment  (Coordination sphere → Cn detection → z-alignment)
# ─────────────────────────────────────────────────────────────────────────────

def detect_coord_sphere(elements: List[str], coords: np.ndarray,
                        metal_idx: int, scale: float = 1.25) -> List[int]:
    """
    Return 0-based indices of atoms in the first coordination sphere of
    the metal (metal_idx), using covalent-radius distance cutoff.
    """
    m_pos = coords[metal_idx]
    m_el  = elements[metal_idx]
    rm    = covalent_radii.get(m_el, covalent_radii["default"])
    sphere = []
    for i, (el, pos) in enumerate(zip(elements, coords)):
        if i == metal_idx:
            continue
        rl  = covalent_radii.get(el, covalent_radii["default"])
        if np.linalg.norm(pos - m_pos) < scale * (rm + rl):
            sphere.append(i)
    return sphere


def _rotation_matrix_to_z(axis: np.ndarray) -> np.ndarray:
    """
    Return 3×3 rotation matrix R such that R @ axis ≈ [0,0,1].
    Uses Rodrigues formula; handles the axis≈−z edge case.
    """
    axis = axis / (np.linalg.norm(axis) + 1e-15)
    z    = np.array([0.0, 0.0, 1.0])
    dot  = float(np.dot(axis, z))
    if abs(dot - 1.0) < 1e-9:            # already z
        return np.eye(3)
    if abs(dot + 1.0) < 1e-9:            # antiparallel → flip around x
        return np.diag([1.0, -1.0, -1.0])
    cross = np.cross(axis, z)
    s     = np.linalg.norm(cross)
    c     = dot
    K     = np.array([[     0, -cross[2],  cross[1]],
                      [ cross[2],      0, -cross[0]],
                      [-cross[1],  cross[0],      0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s + 1e-15))


def _test_cn_axis(vecs: np.ndarray, axis: np.ndarray,
                  n: int, tol_deg: float = 8.0) -> bool:
    """
    Test whether `axis` is an approximate Cn rotation axis for the set of
    unit direction vectors `vecs` (shape N×3, already normalised).
    A Cn rotation maps each vec to another vec in the set.
    """
    angle = 2 * np.pi / n
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    axis = axis / (np.linalg.norm(axis) + 1e-15)
    K  = np.array([[       0, -axis[2],  axis[1]],
                   [ axis[2],        0, -axis[0]],
                   [-axis[1],  axis[0],        0]])
    R  = (cos_a * np.eye(3)
          + sin_a * K
          + (1 - cos_a) * np.outer(axis, axis))
    tol = np.cos(np.radians(tol_deg))
    for v in vecs:
        Rv = R @ v
        # check if Rv is close to any vec in the set
        dots = vecs @ Rv
        if np.max(dots) < tol:
            return False
    return True


def _fibonacci_sphere(n: int = 200) -> np.ndarray:
    """Return n roughly evenly distributed unit vectors on a hemisphere."""
    golden = (1 + 5 ** 0.5) / 2
    pts = []
    for k in range(n):
        theta = np.arccos(1 - (k + 0.5) / n)   # 0 .. pi/2 hemisphere
        phi   = 2 * np.pi * k / golden
        pts.append([np.sin(theta)*np.cos(phi),
                    np.sin(theta)*np.sin(phi),
                    np.cos(theta)])
    return np.array(pts)


def detect_rotation_axis(elements: List[str], coords: np.ndarray,
                         metal_idx: int,
                         coord_sphere: List[int],
                         tol_deg: float = 10.0
                         ) -> Tuple[np.ndarray, int]:
    """
    Detect the principal rotation axis (Cn) from the coordination sphere.

    Strategy
    --------
    1. Build unit direction vectors from metal to each coord-sphere atom.
    2. Generate candidate axes:
       - Atom direction vectors
       - Pair bisectors and cross products (normals)
       - Triplet normals (cross of two pair-vectors)
       - SVD plane-normal and max-variance axis
       - Dense Fibonacci hemisphere grid (robust global search)
    3. Test each candidate for Cn (n = 6,5,4,3,2).
    4. Return highest-order confirmed axis, or SVD fallback (Cn=0).
    """
    m_pos = coords[metal_idx]
    if not coord_sphere:
        return np.array([0.0, 0.0, 1.0]), 0

    vecs = np.array([coords[i] - m_pos for i in coord_sphere], dtype=float)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs  = vecs / (norms + 1e-15)

    n_v = len(vecs)

    # ── SVD axes (always computed) ──
    _, svals, Vt = np.linalg.svd(vecs)
    svd_normal   = Vt[-1]   # smallest variance → plane normal
    svd_major    = Vt[0]    # largest variance

    candidates: List[np.ndarray] = []

    # atom directions
    candidates.extend(list(vecs))

    # pair bisectors, cross products, sum-of-triplet normals
    for a in range(n_v):
        for b in range(a + 1, n_v):
            bis = vecs[a] + vecs[b]
            nb  = np.linalg.norm(bis)
            if nb > 1e-6:
                candidates.append(bis / nb)
            cr  = np.cross(vecs[a], vecs[b])
            nc  = np.linalg.norm(cr)
            if nc > 1e-6:
                candidates.append(cr / nc)
            # also try the negative cross
            for c in range(b + 1, n_v):
                cr2 = np.cross(vecs[a] - vecs[b], vecs[a] - vecs[c])
                nc2 = np.linalg.norm(cr2)
                if nc2 > 1e-6:
                    candidates.append(cr2 / nc2)

    # SVD axes
    candidates.extend([svd_normal, -svd_normal, svd_major, -svd_major])

    # Fibonacci hemisphere grid — catches tilted / lower-symmetry axes
    candidates.extend(list(_fibonacci_sphere(300)))

    # ── score each candidate ──
    best_axis, best_n = svd_normal, 0
    for ax in candidates:
        ax   = np.asarray(ax, dtype=float)
        norm = np.linalg.norm(ax)
        if norm < 1e-9:
            continue
        ax = ax / norm
        for n in [6, 5, 4, 3, 2]:
            if _test_cn_axis(vecs, ax, n, tol_deg):
                if n > best_n:
                    best_n    = n
                    best_axis = ax.copy()
                break

    if best_axis[2] < 0:
        best_axis = -best_axis
    return best_axis, best_n


def apply_alignment(coords: np.ndarray, metal_pos: np.ndarray,
                    axis: np.ndarray) -> np.ndarray:
    """
    Translate so metal_pos → origin, rotate so axis → z,
    then translate back.  Returns transformed coords (N×3).
    """
    R    = _rotation_matrix_to_z(axis)
    rel  = coords - metal_pos
    rot  = (R @ rel.T).T
    return rot + metal_pos


# ─────────────────────────────────────────────────────────────────────────────
# Config persistence  (E1)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "metal_index": "1",
    "use_manual_metal": False,
    "mx": "0.0", "my": "0.0", "mz": "0.0",
    "show_h": True,
    "show_bonds": True,
    "show_atoms": True,
    "show_labels": False,
    "show_displacement": False,
    "proj": "0-180",
    "bond_scale": "1.25",
    "bond_width": "1.0",
    "bond_alpha": "0.30",
    "atom_size": "1.0",
    "use_kabsch": True,
    "last_session_path": "",
    "window_geometry": "1280x800",
    "sash_pos": 820,
}


def load_config() -> dict:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return {**DEFAULT_CONFIG, **cfg}
    except Exception:
        return dict(DEFAULT_CONFIG)


def save_config(cfg: dict):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic colour palette  (B2)
# ─────────────────────────────────────────────────────────────────────────────

def _palette_color(index: int, total: int) -> str:
    """Return a hex colour from a qualitative colormap."""
    cmap = matplotlib.colormaps["tab10"]
    rgba = cmap(index % cmap.N)
    return "#{:02x}{:02x}{:02x}".format(
        int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
    )


# ─────────────────────────────────────────────────────────────────────────────
# UI style  (D1)
# ─────────────────────────────────────────────────────────────────────────────

# Style prefix – all plugin-specific styles use "CPV." dot-notation prefix
# so ttk automatically inherits the correct layout while never colliding
# with or overwriting the host application's ttk styles.
_P = "CPV."

def _apply_style(standalone: bool = True):
    """
    Register plugin-specific ttk styles under the "CPV." dot-notation prefix.

    Plugin mode  (standalone=False):
        - Does NOT call theme_use() – preserves host app theme
        - Does NOT modify any base widget styles (TButton, TFrame, etc.)
        - Only registers named P_* styles

    Standalone mode (standalone=True):
        - Additionally calls theme_use("clam") for a clean look
        - Configures sensible defaults for base widgets
    """
    style = ttk.Style()

    BG   = "#F4F5F7"
    PNL  = "#EAECF0"
    ACC  = "#2563EB"
    TXT  = "#1E2330"
    SUB  = "#5A6072"
    SEP  = "#D0D3DA"

    if standalone:
        # Only touch global styles in standalone mode
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure(".",           background=BG, foreground=TXT,
                        font=("Segoe UI", 9))
        style.configure("TFrame",      background=BG)
        style.configure("TLabel",      background=BG, foreground=TXT)
        style.configure("TCheckbutton",background=PNL, foreground=TXT)
        style.configure("TCombobox",   fieldbackground="white", foreground=TXT)
        style.configure("TLabelframe", background=PNL, foreground=TXT,
                        relief="flat", borderwidth=1)
        style.configure("TLabelframe.Label", background=PNL, foreground=SUB,
                        font=("Segoe UI", 8, "bold"))
        style.configure("TButton",     background="#D6D9E0", foreground=TXT,
                        relief="raised", padding=(6, 4), borderwidth=2)
        style.map("TButton",
                  background=[("active","#BFC3CC"),("pressed","#ADB1BA")],
                  relief=[("pressed","sunken")])

    # ── P_* styles: safe in both modes (only add, never modify host styles) ──
    style.configure(f"{_P}TFrame",      background=BG)
    style.configure(f"{_P}Panel.TFrame",background=PNL)

    style.configure(f"{_P}TLabel",      background=BG, foreground=TXT,
                    font=("Segoe UI", 9))
    style.configure(f"{_P}Header.TLabel",background=BG, foreground=TXT,
                    font=("Segoe UI", 12, "bold"))
    style.configure(f"{_P}Sub.TLabel",  background=BG, foreground=SUB,
                    font=("Segoe UI", 9))
    style.configure(f"{_P}Status.TLabel",background="#DDE0E8", foreground=SUB,
                    font=("Segoe UI", 8), padding=(6, 2))

    style.configure(f"{_P}TLabelframe", background=PNL, foreground=TXT,
                    relief="flat", borderwidth=1, bordercolor=SEP)
    style.configure(f"{_P}TLabelframe.Label", background=PNL, foreground=SUB,
                    font=("Segoe UI", 8, "bold"))

    style.configure(f"{_P}TCheckbutton",background=PNL, foreground=TXT)
    style.configure(f"{_P}TCombobox",   fieldbackground="white", foreground=TXT)

    style.configure(f"{_P}TButton",     background="#D6D9E0", foreground=TXT,
                    relief="raised", padding=(6, 4), borderwidth=2)
    style.map(f"{_P}TButton",
              background=[("active","#BFC3CC"),("pressed","#ADB1BA")],
              relief=[("pressed","sunken")])

    style.configure(f"{_P}Accent.TButton",background=ACC, foreground="white",
                    font=("Segoe UI", 9, "bold"),
                    relief="raised", padding=(6, 4), borderwidth=2)
    style.map(f"{_P}Accent.TButton",
              background=[("active","#1d4ed8"),("pressed","#1e40af")],
              relief=[("pressed","sunken")])

    style.configure(f"{_P}Danger.TButton",background="#DC2626", foreground="white",
                    relief="raised", padding=(2, 1),
                    font=("Segoe UI", 8, "bold"), borderwidth=1)
    style.map(f"{_P}Danger.TButton",
              background=[("active","#b91c1c"),("pressed","#991b1b")],
              relief=[("pressed","sunken")])

    style.configure(f"{_P}Redraw.TButton",background="#16a34a", foreground="white",
                    font=("Segoe UI", 9, "bold"),
                    relief="raised", padding=(6, 4), borderwidth=2)
    style.map(f"{_P}Redraw.TButton",
              background=[("active","#15803d"),("pressed","#166534")],
              relief=[("pressed","sunken")])

    return BG, PNL, ACC, TXT, SUB, SEP


# ─────────────────────────────────────────────────────────────────────────────
# Main viewer class
# ─────────────────────────────────────────────────────────────────────────────

class CoordinationPolarViewer:
    def __init__(self, master: Optional[tk.Misc] = None):
        self._plugin_mode = master is not None
        if master is None:
            self.root = tk.Tk()
            self.root.title("Coordination Polar Viewer")
            self._standalone = True
        else:
            self.root = tk.Toplevel(master)
            self.root.title("Coordination Polar Viewer")
            self._standalone = False

        self._cfg = load_config()
        self.root.geometry(self._cfg.get("window_geometry", "1280x800"))
        self.root.minsize(900, 620)

        BG, PNL, ACC, TXT, SUB, SEP = _apply_style(standalone=self._standalone)
        self._colors = {"bg": BG, "pnl": PNL, "acc": ACC, "txt": TXT, "sub": SUB, "sep": SEP}

        # ── data ──
        self.structures: List[XYZStructure] = []
        self.reference_index = 0

        # ── tk vars ──
        self.show_h_var           = tk.BooleanVar(value=self._cfg["show_h"])
        self.show_bonds_var       = tk.BooleanVar(value=self._cfg["show_bonds"])
        self.show_atoms_var       = tk.BooleanVar(value=self._cfg["show_atoms"])
        self.show_labels_var      = tk.BooleanVar(value=self._cfg["show_labels"])
        self.show_displacement_var= tk.BooleanVar(value=self._cfg["show_displacement"])
        self.use_kabsch_var       = tk.BooleanVar(value=self._cfg["use_kabsch"])

        self.proj_var        = tk.StringVar(value=self._cfg["proj"])
        self.bond_scale_var  = tk.StringVar(value=self._cfg["bond_scale"])
        self.bond_width_var  = tk.StringVar(value=self._cfg.get("bond_width", "1.0"))
        self.bond_alpha_var  = tk.StringVar(value=self._cfg.get("bond_alpha", "0.30"))
        self.atom_size_var   = tk.StringVar(value=self._cfg.get("atom_size", "1.0"))

        self.status_var = tk.StringVar(value="Ready — load XYZ files to begin.")

        self._structure_vars:  List[tk.BooleanVar] = []
        self._memo_vars:       List[tk.StringVar]  = []
        self._color_buttons:   List[tk.Button]     = []

        # ── scatter artists for tooltip ──
        self._scatter_meta: list = []   # list of (scatter, struct_index, atom_indices)

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ══════════════════════════════════════════════════════════════════════════
    # UI construction
    # ══════════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        C = self._colors

        # ── outer frame ──
        outer = ttk.Frame(self.root, padding=(10, 8, 10, 0))
        outer.pack(fill="both", expand=True)

        # ── header ──
        hdr = ttk.Frame(outer)
        hdr.pack(fill="x", pady=(0, 6))
        ttk.Label(hdr, text="Coordination Polar Viewer", style="CPV.Header.TLabel").pack(side="left")
        ttk.Label(hdr, text="v0.1.2",
                  style="CPV.Sub.TLabel").pack(side="left", padx=(10, 0))

        # ── PanedWindow (D2) ──
        paned = tk.PanedWindow(outer, orient="horizontal",
                               sashwidth=6, sashpad=2,
                               bg=C["sep"], relief="flat")
        paned.pack(fill="both", expand=True)

        left_frame = ttk.Frame(paned)
        right_frame = ttk.Frame(paned, style="CPV.Panel.TFrame")
        paned.add(left_frame,  minsize=520)
        paned.add(right_frame, minsize=280)

        sash = self._cfg.get("sash_pos", 820)
        self.root.after(50, lambda: paned.sash_place(0, sash, 0))
        self._paned = paned

        # ── matplotlib figure ──
        self.fig = plt.Figure(figsize=(6.8, 6.0), dpi=110, facecolor="#F7F8FA")
        self.ax  = self.fig.add_subplot(1, 1, 1, projection="polar")
        self.fig.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.06)

        self.canvas = FigureCanvasTkAgg(self.fig, master=left_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, left_frame)
        toolbar.update()

        # right-click context menu on canvas (C2)
        self._ctx_menu = tk.Menu(self.root, tearoff=0)
        self._ctx_menu.add_command(label="Save PNG…",  command=lambda: self._export("png"))
        self._ctx_menu.add_command(label="Save SVG…",  command=lambda: self._export("svg"))
        self._ctx_menu.add_command(label="Save PDF…",  command=lambda: self._export("pdf"))
        self._ctx_menu.add_separator()
        self._ctx_menu.add_command(label="Export CSV…", command=self._export_csv)
        self._ctx_menu.add_separator()
        self._ctx_menu.add_checkbutton(label="Show displacement arrows",
                                       variable=self.show_displacement_var,
                                       command=self.redraw)
        self.canvas.get_tk_widget().bind("<Button-3>", self._show_context_menu)
        self.canvas.get_tk_widget().bind("<Button-2>", self._show_context_menu)  # macOS

        # ── right panel with notebook ──
        nb = ttk.Notebook(right_frame)
        nb.pack(fill="both", expand=True, padx=6, pady=6)

        ctrl_tab     = ttk.Frame(nb, style="CPV.Panel.TFrame", padding=6)
        analysis_tab = ttk.Frame(nb, style="CPV.Panel.TFrame", padding=6)
        nb.add(ctrl_tab,     text="  Controls  ")
        nb.add(analysis_tab, text="  Analysis  ")

        self._build_controls_tab(ctrl_tab)
        self._build_analysis_tab(analysis_tab)

        # ── status bar (D3) ──
        sb = ttk.Label(self.root, textvariable=self.status_var, style="CPV.Status.TLabel",
                       anchor="w")
        sb.pack(fill="x", side="bottom")

        self.redraw()

    # ── Controls tab ──────────────────────────────────────────────────────────

    def _build_controls_tab(self, parent):
        C = self._colors

        # ── scrollable inner panel ──
        vsb = ttk.Scrollbar(parent, orient="vertical")
        vsb.pack(side="right", fill="y")
        cv = tk.Canvas(parent, yscrollcommand=vsb.set,
                       bg=C["pnl"], highlightthickness=0)
        cv.pack(side="left", fill="both", expand=True)
        vsb.config(command=cv.yview)
        inner = ttk.Frame(cv, style="CPV.Panel.TFrame")
        win = cv.create_window((0, 0), window=inner, anchor="nw")

        def _resize(e):
            cv.configure(scrollregion=cv.bbox("all"))
            cv.itemconfig(win, width=cv.winfo_width())
        inner.bind("<Configure>", _resize)
        cv.bind("<Configure>", _resize)
        cv.bind_all("<MouseWheel>",
                    lambda e: cv.yview_scroll(-1*(e.delta//120), "units"))

        P = C["pnl"]  # panel bg shorthand

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # FILES
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━
        fb = ttk.LabelFrame(inner, text="XYZ Files", padding=(6, 4), style="CPV.TLabelframe")
        fb.pack(fill="x", pady=(0, 6), padx=2)

        btn_row = ttk.Frame(fb, style="CPV.Panel.TFrame")
        btn_row.pack(fill="x", pady=(0, 4))
        ttk.Button(btn_row, text="Load Ref…", style="CPV.Accent.TButton",
                   command=self.load_reference).pack(side="left", expand=True, fill="x", padx=(0, 2))
        ttk.Button(btn_row, text="Add Comp…",
                   command=self.add_comparison, style="CPV.TButton").pack(side="left", expand=True, fill="x", padx=(2, 2))
        ttk.Button(btn_row, text="Clear",
                   command=self.clear_all, style="CPV.TButton").pack(side="left", expand=True, fill="x", padx=(2, 0))

        session_row = ttk.Frame(fb, style="CPV.Panel.TFrame")
        session_row.pack(fill="x", pady=(0, 4))
        ttk.Button(session_row, text="Save Session…",
                   command=self.save_session_dialog, style="CPV.TButton").pack(side="left", expand=True, fill="x", padx=(0, 2))
        ttk.Button(session_row, text="Load Session…",
                   command=self.load_session_dialog, style="CPV.TButton").pack(side="left", expand=True, fill="x", padx=(2, 0))

        self.struct_frame = ttk.Frame(fb, style="CPV.Panel.TFrame")
        self.struct_frame.pack(fill="x")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # DISPLAY
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━
        db = ttk.LabelFrame(inner, text="Display", padding=(6, 4), style="CPV.TLabelframe")
        db.pack(fill="x", pady=(0, 6), padx=2)

        # 3-column checkbox grid (2 rows × 3 cols)
        check_items = [
            ("H atoms",      self.show_h_var),
            ("Bonds",        self.show_bonds_var),
            ("Atoms",        self.show_atoms_var),
            ("Labels",       self.show_labels_var),
            ("Displace.",    self.show_displacement_var),
            ("Kabsch RMSD",  self.use_kabsch_var),
        ]
        cgrid = ttk.Frame(db, style="CPV.Panel.TFrame")
        cgrid.pack(fill="x", pady=(0, 4))
        for k, (txt, var) in enumerate(check_items):
            ttk.Checkbutton(cgrid, text=txt, variable=var,
                            command=self.redraw, style="CPV.TCheckbutton").grid(
                row=k // 3, column=k % 3, sticky="w", padx=4, pady=1)

        # ── Options row: Projection | Bond scale | Bond width | Bond alpha | Atom size ──
        opt = ttk.Frame(db, style="CPV.Panel.TFrame")
        opt.pack(fill="x", pady=(0, 4))

        ttk.Label(opt, text="Proj:", background=P,
                  font=("Segoe UI", 8)).pack(side="left")
        ttk.Combobox(opt, textvariable=self.proj_var,
                     values=("0-180", "0-90 folded"),
                     state="readonly", width=10,
                     font=("Segoe UI", 8)).pack(side="left", padx=(2, 8))
        self.proj_var.trace_add("write", lambda *_: self.redraw())

        for lbl, var, w in [("B.scale", self.bond_scale_var, 4),
                             ("B.width", self.bond_width_var,  4),
                             ("B.alpha", self.bond_alpha_var,  4),
                             ("Atom sz", self.atom_size_var,   4)]:
            ttk.Label(opt, text=lbl, background=P,
                      font=("Segoe UI", 8)).pack(side="left", padx=(4, 1))
            ttk.Entry(opt, textvariable=var, width=w,
                      font=("Segoe UI", 8)).pack(side="left")

        # ── Redraw + Export row ──
        act = ttk.Frame(db, style="CPV.Panel.TFrame")
        act.pack(fill="x", pady=(4, 0))
        ttk.Button(act, text="⟳  Redraw",
                   style="CPV.Redraw.TButton",
                   command=self.redraw).pack(side="left", expand=True, fill="x", padx=(0, 4))
        for fmt in ("PNG", "SVG", "PDF", "CSV"):
            cmd = self._export_csv if fmt == "CSV" else (lambda f=fmt.lower(): self._export(f))
            ttk.Button(act, text=fmt, command=cmd, style="CPV.TButton").pack(
                side="left", expand=True, fill="x", padx=1)

    # ── Analysis tab (A3) ─────────────────────────────────────────────────────

    def _build_analysis_tab(self, parent):
        top = ttk.Frame(parent)
        top.pack(fill="x", pady=(0, 4))
        ttk.Button(top, text="⟳  Refresh",
                   style="CPV.Redraw.TButton",
                   command=self._refresh_analysis).pack(side="left")
        ttk.Button(top, text="Export TXT…",
                   command=self._export_analysis_report, style="CPV.TButton").pack(side="left", padx=(4, 0))
        ttk.Button(top, text="Export CSV…",
                   command=self._export_analysis_csv, style="CPV.TButton").pack(side="left", padx=(4, 0))
        ttk.Button(top, text="📈 ML plot…",
                   command=self._plot_ml_trends, style="CPV.TButton").pack(side="left", padx=(4, 0))
        ttk.Button(top, text="📈 Ionic radii plot…",
                   command=self._plot_ionic_radii, style="CPV.TButton").pack(side="left", padx=(4, 0))
        ttk.Button(top, text="📈 Polar dist…",
                   command=self._plot_polar_dist, style="CPV.TButton").pack(side="left", padx=(4, 0))

        # sub-notebook: each analysis section as a tab
        self._analysis_nb = ttk.Notebook(parent)
        self._analysis_nb.pack(fill="both", expand=True)

        def _make_text_tab(title):
            frm = ttk.Frame(self._analysis_nb)
            self._analysis_nb.add(frm, text=title)
            vsb = ttk.Scrollbar(frm)
            vsb.pack(side="right", fill="y")
            hsb = ttk.Scrollbar(frm, orient="horizontal")
            hsb.pack(side="bottom", fill="x")
            txt = tk.Text(frm, wrap="none",
                          font=("Courier New", 9),
                          bg="#F8F9FB", fg="#1E2330",
                          relief="flat", borderwidth=1,
                          state="disabled",
                          yscrollcommand=vsb.set,
                          xscrollcommand=hsb.set)
            txt.pack(fill="both", expand=True)
            vsb.config(command=txt.yview)
            hsb.config(command=txt.xview)
            return txt

        self._txt_rmsd     = _make_text_tab("RMSD")
        self._txt_ml       = _make_text_tab("M–L Δ")
        self._txt_geom     = _make_text_tab("Geometry")
        self._txt_trans    = _make_text_tab("Trans")
        self._txt_polar    = _make_text_tab("Polar stats")
        self._txt_shannon  = _make_text_tab("Ionic radii")

        # keep legacy attribute pointing to RMSD tab for compat
        self.analysis_text = self._txt_rmsd

    # ══════════════════════════════════════════════════════════════════════════
    # Structure list widget
    # ══════════════════════════════════════════════════════════════════════════

    def _remove_structure(self, idx: int):
        if idx >= len(self.structures):
            return
        self.structures.pop(idx)
        if self.reference_index >= len(self.structures):
            self.reference_index = max(0, len(self.structures) - 1)
        self._refresh_structure_list()
        self.redraw()
        self._set_status(f"Removed structure. {len(self.structures)} remaining.")

    def _set_structure_visible(self, idx: int, var: tk.BooleanVar):
        if idx < len(self.structures):
            self.structures[idx].visible = var.get()
        self.redraw()

    def _metal_options_for(self, st) -> List[str]:
        opts = []
        for i, el in enumerate(st.elements):
            if el in TRANSITION_METALS:
                opts.append(f"{i+1}: {el}")
        if not opts:
            for i, el in enumerate(st.elements):
                opts.append(f"{i+1}: {el}")
        return opts

    def _metal_option_str(self, st) -> str:
        idx = st.metal_index
        if 0 <= idx < len(st.elements):
            return f"{idx+1}: {st.elements[idx]}"
        return f"{idx+1}"

    def _on_struct_metal_select(self, struct_idx: int, sv: tk.StringVar, idx_var: tk.StringVar):
        val = sv.get()
        try:
            atom_idx = int(val.split(":")[0].strip()) - 1
            self.structures[struct_idx].metal_index = atom_idx
            idx_var.set(str(atom_idx + 1))
            self.redraw()
        except (ValueError, IndexError):
            pass

    def _on_struct_idx_entry(self, struct_idx: int, idx_var: tk.StringVar):
        try:
            atom_idx = int(idx_var.get()) - 1
            st = self.structures[struct_idx]
            if 0 <= atom_idx < len(st.elements):
                st.metal_index = atom_idx
                # update manual xyz to match
                xyz = st.coords[atom_idx]
                st.mx, st.my, st.mz = float(xyz[0]), float(xyz[1]), float(xyz[2])
                self.redraw()
        except (ValueError, IndexError):
            pass

    def _refresh_structure_list(self):
        for w in self.struct_frame.winfo_children():
            w.destroy()
        self._structure_vars.clear()
        self._memo_vars.clear()
        self._color_buttons.clear()

        BG = "#D8DBE2"
        FG = "#3A3D44"
        FONT8 = ("Segoe UI", 8)

        for i, st in enumerate(self.structures):
            var = tk.BooleanVar(value=getattr(st, "visible", True))
            memo_var = tk.StringVar(value=st.memo)
            self._structure_vars.append(var)
            self._memo_vars.append(memo_var)

            card = tk.Frame(self.struct_frame, bg=BG, relief="groove", borderwidth=1)
            card.pack(fill="x", pady=2, padx=2)

            # ── Row A: [colour] ☑ label  [note entry]  [x] ──
            ra = tk.Frame(card, bg=BG)
            ra.pack(fill="x", padx=4, pady=(3, 1))

            cb = tk.Button(ra, bg=st.color, width=2, height=1,
                           relief="raised", borderwidth=1, cursor="hand2",
                           command=lambda idx=i: self._pick_color(idx))
            cb.pack(side="left", padx=(0, 3))
            self._color_buttons.append(cb)

            tag = " [ref]" if i == self.reference_index else ""
            ttk.Checkbutton(ra, text=f"{i+1}. {st.label}{tag}",
                            variable=var,
                            command=lambda idx=i, bv=var: self._set_structure_visible(idx, bv),
                            style="CPV.TCheckbutton").pack(side="left")

            ttk.Button(ra, text="✕", width=2, style="CPV.Danger.TButton",
                       command=lambda idx=i: self._remove_structure(idx)
                       ).pack(side="right", padx=(2, 0))

            memo_entry = ttk.Entry(ra, textvariable=memo_var, width=8, font=FONT8)
            memo_entry.pack(side="right", padx=(2, 2))
            tk.Label(ra, text="note:", bg=BG, font=FONT8, fg=FG).pack(side="right")
            memo_entry.bind("<Return>",
                            lambda e, idx=i, mv=memo_var: self._set_memo(idx, mv))
            memo_entry.bind("<FocusOut>",
                            lambda e, idx=i, mv=memo_var: self._set_memo(idx, mv))

            # ── Row B: Metal [dropdown]  idx [entry]  ☐ManualXYZ  x[..] y[..] z[..] ──
            rb = tk.Frame(card, bg=BG)
            rb.pack(fill="x", padx=4, pady=(0, 3))

            tk.Label(rb, text="Metal:", bg=BG, font=FONT8, fg=FG).pack(side="left")
            metal_options = self._metal_options_for(st)
            metal_str_var = tk.StringVar(value=self._metal_option_str(st))
            idx_var = tk.StringVar(value=str(st.metal_index + 1))

            metal_dd = ttk.Combobox(rb, textvariable=metal_str_var,
                                    values=metal_options, state="readonly",
                                    width=8, font=FONT8)
            metal_dd.pack(side="left", padx=(1, 3))
            metal_dd.bind("<<ComboboxSelected>>",
                          lambda e, idx=i, sv=metal_str_var, iv=idx_var:
                              self._on_struct_metal_select(idx, sv, iv))

            tk.Label(rb, text="Atom #:", bg=BG, font=FONT8, fg=FG).pack(side="left")
            idx_entry = ttk.Entry(rb, textvariable=idx_var, width=3, font=FONT8)
            idx_entry.pack(side="left", padx=(1, 4))
            idx_entry.bind("<Return>",
                           lambda e, idx=i, iv=idx_var: self._on_struct_idx_entry(idx, iv))
            idx_entry.bind("<FocusOut>",
                           lambda e, idx=i, iv=idx_var: self._on_struct_idx_entry(idx, iv))

            manual_var = tk.BooleanVar(value=st.use_manual)
            mx_var = tk.StringVar(value=f"{st.mx:.3f}")
            my_var = tk.StringVar(value=f"{st.my:.3f}")
            mz_var = tk.StringVar(value=f"{st.mz:.3f}")

            def make_upd(idx, mx, my, mz):
                def _u(*_):
                    try:
                        self.structures[idx].mx = float(mx.get())
                        self.structures[idx].my = float(my.get())
                        self.structures[idx].mz = float(mz.get())
                        self.redraw()
                    except ValueError:
                        pass
                return _u

            upd = make_upd(i, mx_var, my_var, mz_var)
            xyz_entries = []

            for lbl, xv in [("x", mx_var), ("y", my_var), ("z", mz_var)]:
                tk.Label(rb, text=lbl, bg=BG, font=FONT8, fg=FG).pack(side="left")
                e = ttk.Entry(rb, textvariable=xv, width=5, font=FONT8)
                e.pack(side="left", padx=(1, 2))
                e.bind("<Return>",   lambda ev, u=upd: u())
                e.bind("<FocusOut>", lambda ev, u=upd: u())
                if not st.use_manual:
                    e.configure(state="disabled")
                xyz_entries.append(e)

            def make_xyz_frame_ref(entries):
                return entries

            entries_ref = make_xyz_frame_ref(xyz_entries)

            def toggle_manual(idx=i, bv=manual_var, ents=entries_ref):
                self.structures[idx].use_manual = bv.get()
                state = "normal" if bv.get() else "disabled"
                for e in ents:
                    try:
                        e.configure(state=state)
                    except Exception:
                        pass
                self.redraw()

            ttk.Checkbutton(rb, text="Manual coord", variable=manual_var,
                            command=toggle_manual, style="CPV.TCheckbutton").pack(side="left", padx=(2, 0))

            # ── Row C: Axis alignment ──
            rc = tk.Frame(card, bg=BG)
            rc.pack(fill="x", padx=4, pady=(0, 4))

            align_var = tk.BooleanVar(value=st.use_align)

            def toggle_align(idx=i, av=align_var):
                self.structures[idx].use_align = av.get()
                self.redraw()

            ttk.Checkbutton(rc, text="Align axis→z",
                            variable=align_var,
                            command=toggle_align, style="CPV.TCheckbutton").pack(side="left")

            # Cn label
            if st.align_axis is not None:
                cn_txt = (f"C{st.align_Cn}" if st.align_Cn >= 2 else "SVD")
                ax     = st.align_axis
                cn_str = (f"{cn_txt}  [{ax[0]:.2f}, {ax[1]:.2f}, {ax[2]:.2f}]")
            else:
                cn_str = "not detected"
            cn_label = tk.Label(rc, text=cn_str, bg=BG,
                                font=("Segoe UI", 7), fg="#2563EB")
            cn_label.pack(side="left", padx=(4, 0))

            ttk.Button(rc, text="Auto-align",
                       command=lambda idx=i: self._run_alignment(idx, style="CPV.TButton")
                       ).pack(side="right", padx=(4, 0))
            ttk.Button(rc, text="Save aligned…",
                       command=lambda idx=i: self._save_aligned_xyz(idx, style="CPV.TButton")
                       ).pack(side="right", padx=(4, 0))
            ttk.Button(rc, text="Coord sphere…",
                       command=lambda idx=i: self._open_coord_sphere_editor(idx, style="CPV.TButton")
                       ).pack(side="right")

    # ══════════════════════════════════════════════════════════════════════════
    # Axis alignment helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _run_alignment(self, struct_idx: int):
        """
        Auto-detect coordination sphere + Cn axis for structures[struct_idx],
        store results in the XYZStructure, then redraw.
        """
        if struct_idx >= len(self.structures):
            return
        st = self.structures[struct_idx]
        metal_pos = self._get_metal_position(st)
        if metal_pos is None:
            return

        # detect coord sphere if not already manually set
        if st.coord_sphere is None:
            st.coord_sphere = detect_coord_sphere(
                st.elements, st.coords, st.metal_index)

        axis, cn = detect_rotation_axis(
            st.elements, st.coords, st.metal_index, st.coord_sphere)

        st.align_axis = axis
        st.align_Cn   = cn
        st.use_align  = True

        label = f"C{cn}" if cn >= 2 else "SVD"
        self._set_status(
            f"Aligned {st.label}: {label} axis = "
            f"[{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}]"
        )
        self._refresh_structure_list()
        self.redraw()

    def _save_aligned_xyz(self, struct_idx: int):
        """
        Export the aligned coordinates of structures[struct_idx] as an XYZ file.
        The metal atom is translated to the origin and coordinates are rotated
        so that the detected/manual axis aligns with z.
        If use_align is False or no axis is set, saves the raw coordinates
        (metal-centred translation only).
        """
        if struct_idx >= len(self.structures):
            return
        st = self.structures[struct_idx]

        metal_pos = self._get_metal_position(st)
        if metal_pos is None:
            messagebox.showerror("Save aligned XYZ",
                                 "No metal position defined for this structure.")
            return

        # Build transformed coords
        coords = np.asarray(st.coords, dtype=float)

        if st.use_align and st.align_axis is not None and                 np.linalg.norm(st.align_axis) > 1e-9:
            aligned = apply_alignment(coords, metal_pos, st.align_axis)
            # Translate so metal is at origin
            metal_new = apply_alignment(metal_pos.reshape(1, 3),
                                        metal_pos, st.align_axis)[0]
            aligned   = aligned - metal_new
            note = (f"Cn=C{st.align_Cn}" if st.align_Cn >= 2 else "SVD-aligned")
            ax   = st.align_axis
            comment = (f"auto-aligned {note}  axis=[{ax[0]:.4f},{ax[1]:.4f},{ax[2]:.4f}]"
                       f"  source={st.path}")
        else:
            # Just centre on metal
            aligned = coords - metal_pos
            comment = f"metal-centred (no axis alignment)  source={st.path}"

        # Ask for save path
        default_name = os.path.splitext(os.path.basename(st.path))[0] + "_aligned.xyz"
        path = filedialog.asksaveasfilename(
            title="Save aligned XYZ",
            initialfile=default_name,
            defaultextension=".xyz",
            filetypes=[("XYZ files", "*.xyz"), ("All files", "*.*")],
        )
        if not path:
            return

        # Write XYZ
        n = len(st.elements)
        lines = [str(n), comment]
        for el, pos in zip(st.elements, aligned):
            lines.append(f"{el:<4s}  {pos[0]:14.8f}  {pos[1]:14.8f}  {pos[2]:14.8f}")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        self._set_status(f"Saved aligned XYZ: {os.path.basename(path)}")

    def _open_coord_sphere_editor(self, struct_idx: int):
        """
        Open a Toplevel dialog showing coordination sphere atoms with
        checkboxes so the user can include/exclude atoms before re-running
        alignment.
        """
        if struct_idx >= len(self.structures):
            return
        st = self.structures[struct_idx]
        if st.coord_sphere is None:
            st.coord_sphere = detect_coord_sphere(
                st.elements, st.coords, st.metal_index)

        win = tk.Toplevel(self.root)
        win.title(f"Coordination Sphere — {st.label}")
        win.geometry("340x420")
        win.resizable(True, True)
        win.grab_set()

        ttk.Label(win, text="Edit coordination sphere atoms.",
                  font=("Segoe UI", 9)).pack(anchor="w", padx=10, pady=(8, 2))
        ttk.Label(win,
                  text="Check atoms to include; uncheck to exclude from axis detection.",
                  font=("Segoe UI", 8), foreground="#5A6072",
                  wraplength=310).pack(anchor="w", padx=10)

        # metal info
        m_el = st.elements[st.metal_index] if st.metal_index < len(st.elements) else "?"
        metal_pos = self._get_metal_position(st)
        ttk.Label(win,
                  text=f"Metal: #{st.metal_index+1} {m_el}",
                  font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=10, pady=(6, 2))

        # scrollable atom list
        frm = ttk.Frame(win)
        frm.pack(fill="both", expand=True, padx=10, pady=4)
        vsb = ttk.Scrollbar(frm)
        vsb.pack(side="right", fill="y")
        lb_canvas = tk.Canvas(frm, yscrollcommand=vsb.set, highlightthickness=0)
        lb_canvas.pack(side="left", fill="both", expand=True)
        vsb.config(command=lb_canvas.yview)
        inner = ttk.Frame(lb_canvas)
        lb_canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: lb_canvas.configure(scrollregion=lb_canvas.bbox("all")))

        # full_sphere = all atoms bonded to metal at scale 1.5 (wide net)
        full_sphere = detect_coord_sphere(
            st.elements, st.coords, st.metal_index, scale=1.5)
        current_set = set(st.coord_sphere)
        check_vars: List[Tuple[int, tk.BooleanVar]] = []

        for atom_i in sorted(full_sphere):
            el   = st.elements[atom_i]
            pos  = st.coords[atom_i]
            dist = float(np.linalg.norm(pos - metal_pos)) if metal_pos is not None else 0.0
            v    = tk.BooleanVar(value=(atom_i in current_set))
            check_vars.append((atom_i, v))
            row  = ttk.Frame(inner)
            row.pack(fill="x", pady=1)
            ttk.Checkbutton(row, variable=v, style="CPV.TCheckbutton").pack(side="left")
            ttk.Label(row,
                      text=f"#{atom_i+1:>3}  {el:<3}  {dist:6.3f} Å"
                      ).pack(side="left")

        def _apply():
            st.coord_sphere = [i for i, v in check_vars if v.get()]
            win.destroy()
            self._run_alignment(struct_idx)

        def _reset():
            st.coord_sphere = None
            win.destroy()
            self._run_alignment(struct_idx)

        btn_row = ttk.Frame(win)
        btn_row.pack(fill="x", padx=10, pady=(4, 8))
        ttk.Button(btn_row, text="Apply & Re-align",
                   style="CPV.Accent.TButton",
                   command=_apply).pack(side="left", expand=True, fill="x", padx=(0, 4))
        ttk.Button(btn_row, text="Reset to auto",
                   command=_reset, style="CPV.TButton").pack(side="left", expand=True, fill="x")

    def _set_memo(self, idx: int, var: tk.StringVar):
        if idx < len(self.structures):
            self.structures[idx].memo = var.get()
            self.redraw()

    def _pick_color(self, idx: int):
        if idx >= len(self.structures):
            return
        current = self.structures[idx].color
        result = colorchooser.askcolor(color=current, title=f"Color for structure {idx+1}")
        if result and result[1]:
            self.structures[idx].color = result[1]
            self._color_buttons[idx].configure(bg=result[1])
            self.redraw()

    # ══════════════════════════════════════════════════════════════════════════
    # Metal auto-detect (A2)
    # ══════════════════════════════════════════════════════════════════════════

    # ══════════════════════════════════════════════════════════════════════════
    # File actions
    # ══════════════════════════════════════════════════════════════════════════

    def load_reference(self):
        path = filedialog.askopenfilename(
            title="Load reference XYZ",
            filetypes=[("XYZ files", "*.xyz"), ("All files", "*.*")],
        )
        if not path:
            return
        self._load_xyz_async(path, is_reference=True)

    def add_comparison(self):
        paths = filedialog.askopenfilenames(
            title="Add comparison XYZ",
            filetypes=[("XYZ files", "*.xyz"), ("All files", "*.*")],
        )
        if not paths:
            return
        for path in paths:
            self._load_xyz_async(path, is_reference=False)

    def _load_xyz_async(self, path: str, is_reference: bool):
        """Load XYZ in a background thread to avoid UI freeze (E3)."""
        self._set_status("Loading…")

        def _worker():
            try:
                elements, coords = load_xyz(path)
            except Exception as exc:
                self.root.after(0, lambda: messagebox.showerror("XYZ load error", str(exc)))
                self.root.after(0, lambda: self._set_status("Load failed."))
                return

            label = os.path.splitext(os.path.basename(path))[0]
            n = len(self.structures) if not is_reference else 0
            color = _palette_color(n, max(n + 1, 1))
            # auto-detect metal index for this structure
            auto_metal = 0
            for mi, el in enumerate(elements):
                if el in TRANSITION_METALS:
                    auto_metal = mi
                    break
            st = XYZStructure(label=label, path=path,
                              elements=elements, coords=coords,
                              color=color, metal_index=auto_metal)

            def _apply():
                if is_reference:
                    if self.structures:
                        self.structures[0] = st
                    else:
                        self.structures.append(st)
                    self.reference_index = 0
                else:
                    self.structures.append(st)
                self._refresh_structure_list()
                self.redraw()
                self._set_status(
                    f"Loaded {'reference' if is_reference else 'comparison'}: {label}  "
                    f"({len(elements)} atoms)  —  {len(self.structures)} structure(s) total."
                )

            self.root.after(0, _apply)

        threading.Thread(target=_worker, daemon=True).start()

    def clear_all(self):
        self.structures.clear()
        self._refresh_structure_list()
        self.redraw()
        self._set_status("Cleared all structures.")

    # ══════════════════════════════════════════════════════════════════════════
    # Metal helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _get_metal_position(self, st=None) -> Optional[np.ndarray]:
        """Return the metal origin for a structure, using its own fields."""
        if not self.structures:
            return None
        target = st if st is not None else self.structures[self.reference_index]
        # Manual XYZ mode for this structure
        if getattr(target, "use_manual", False):
            return np.array([target.mx, target.my, target.mz], dtype=float)
        # Atom index mode
        idx = getattr(target, "metal_index", 0)
        if 0 <= idx < len(target.coords):
            return target.coords[idx].copy()
        return target.coords[0].copy()

    # ══════════════════════════════════════════════════════════════════════════
    # Polar coordinate transform
    # ══════════════════════════════════════════════════════════════════════════

    def _polar_coords(self, coords: np.ndarray, metal_pos: np.ndarray,
                     st=None):
        """
        Convert Cartesian coords to polar (theta, r) centred on metal_pos.
        If st.use_align and st.align_axis is set, rotate coords so that
        the alignment axis coincides with z before computing theta.
        """
        c = np.asarray(coords, dtype=float)
        if st is not None and getattr(st, "use_align", False):
            axis = getattr(st, "align_axis", None)
            if axis is not None and np.linalg.norm(axis) > 1e-9:
                c = apply_alignment(c, metal_pos, axis)
        rel   = c - metal_pos
        r     = np.linalg.norm(rel, axis=1)
        theta = np.where(r > 1e-9,
                         np.arccos(np.clip(rel[:, 2] / (r + 1e-15), -1.0, 1.0)),
                         0.0)
        if self.proj_var.get() == "0-90 folded":
            theta = np.minimum(theta, np.pi - theta)
        return theta, r

    def _metal_atom_index(self, st=None) -> int:
        """Return the 0-based metal atom index for st (or reference if None)."""
        target = st
        if target is None and self.structures:
            target = self.structures[self.reference_index]
        if target is not None:
            return getattr(target, "metal_index", 0)
        return 0

    def _visible_indices(self, elements: List[str], st=None) -> List[int]:
        metal_idx = self._metal_atom_index(st)
        if self.show_h_var.get():
            return [i for i in range(len(elements)) if i != metal_idx]
        return [i for i, el in enumerate(elements) if el != "H" and i != metal_idx]

    def _bond_scale(self) -> float:
        try:
            return max(0.5, float(self.bond_scale_var.get()))
        except ValueError:
            return 1.25

    def _bond_alpha(self) -> float:
        """Return bond transparency clipped to the Matplotlib alpha range [0, 1]."""
        try:
            return min(1.0, max(0.0, float(self.bond_alpha_var.get())))
        except ValueError:
            return 0.30

    # ══════════════════════════════════════════════════════════════════════════
    # Main redraw
    # ══════════════════════════════════════════════════════════════════════════

    def redraw(self):
        self.ax.clear()
        self._scatter_meta.clear()

        if not self.structures:
            self.ax.set_title("Load one or more XYZ files")
            self.ax.set_theta_zero_location("N")
            self.ax.set_theta_direction(-1)
            self.ax.set_thetamin(0)
            self.ax.set_thetamax(180)
            self.canvas.draw_idle()
            return

        # Validate using reference structure (for error messages)
        if self._get_metal_position() is None:
            return

        legend_handles = []
        plotted = 0

        # ── Metal origin marker (always at r=0, centre of polar plot) ──
        metal_idx = self._metal_atom_index()
        ref_st = self.structures[self.reference_index]
        metal_el = (ref_st.elements[metal_idx]
                    if 0 <= metal_idx < len(ref_st.elements) else "M")
        self.ax.scatter(0, 0,
                        s=120,
                        facecolor=cpk_color(metal_el),
                        edgecolor="#333333",
                        linewidth=1.2,
                        zorder=10,
                        marker="*")
        self.ax.text(0.05, 0.02, metal_el,
                     fontsize=8, color="#333333", zorder=11)
        legend_handles.append(
            Line2D([0], [0], marker="*", color="w",
                   markerfacecolor=cpk_color(metal_el),
                   markeredgecolor="#333333",
                   markersize=10, label=f"{metal_el} (origin)")
        )

        # pre-compute reference polar coords for displacement arrows
        # (reference uses its own metal as origin)
        ref_metal_pos = self._get_metal_position(ref_st)
        ref_th, ref_r = self._polar_coords(ref_st.coords, ref_metal_pos, ref_st)

        for sidx, st in enumerate(self.structures):
            if sidx < len(self._structure_vars):
                st.visible = self._structure_vars[sidx].get()
            if not getattr(st, "visible", True):
                continue

            color = st.color
            # KEY FIX: each structure is centred on its OWN metal atom
            st_metal_pos = self._get_metal_position(st)
            theta, radius = self._polar_coords(st.coords, st_metal_pos, st)
            visible = set(self._visible_indices(st.elements, st))

            memo_suffix = f"  [{st.memo}]" if st.memo else ""
            label = st.label + (" [ref]" if sidx == self.reference_index else "") + memo_suffix

            # ── bonds ──
            if self.show_bonds_var.get():
                bonds = guess_bonds(st.elements, st.coords, scale=self._bond_scale())
                for i, j in bonds:
                    if i not in visible or j not in visible:
                        continue
                    try:
                        bw = max(0.3, float(self.bond_width_var.get()))
                    except ValueError:
                        bw = 1.0
                    ba = self._bond_alpha()
                    self.ax.plot([theta[i], theta[j]], [radius[i], radius[j]],
                                 color=color,
                                 alpha=ba if sidx else ba * 0.75,
                                 linewidth=bw * (1.0 if sidx else 1.2))

            # ── displacement arrows (B1) ──
            if (self.show_displacement_var.get()
                    and sidx != self.reference_index
                    and len(st.coords) == len(ref_st.coords)):
                for i in visible:
                    dt = theta[i] - ref_th[i]
                    dr = radius[i] - ref_r[i]
                    if abs(dt) < 1e-6 and abs(dr) < 1e-6:
                        continue
                    self.ax.annotate(
                        "", xy=(theta[i], radius[i]),
                        xytext=(ref_th[i], ref_r[i]),
                        arrowprops=dict(arrowstyle="->", color=color,
                                        lw=0.9, alpha=0.65),
                    )

            # ── atoms ──
            scat_indices = []
            scat_th, scat_r = [], []
            if self.show_atoms_var.get():
                for i in visible:
                    el = st.elements[i]
                    try:
                        asf = max(0.2, float(self.atom_size_var.get()))
                    except ValueError:
                        asf = 1.0
                    sc = self.ax.scatter(
                        theta[i], radius[i],
                        s=(28 if el != "H" else 12) * asf,
                        facecolor=cpk_color(el),
                        edgecolor=color,
                        linewidth=0.9,
                        alpha=0.88 if sidx else 0.97,
                        zorder=5,
                    )
                    scat_indices.append(i)
                    scat_th.append(theta[i])
                    scat_r.append(radius[i])

                # store for tooltip
                if scat_indices:
                    self._scatter_meta.append((sc, sidx, scat_indices,
                                               scat_th, scat_r))

            # ── atom labels ──
            if self.show_labels_var.get():
                for i in visible:
                    self.ax.text(theta[i], radius[i],
                                 f"{i+1}{st.elements[i]}",
                                 fontsize=7, ha="center", va="center")

            legend_handles.append(
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=color, markersize=8, label=label)
            )
            plotted += 1

        # ── axes ──
        proj = self.proj_var.get()
        self.ax.set_theta_zero_location("N")
        self.ax.set_theta_direction(-1)

        if proj == "0-90 folded":
            # fold: θ runs 0→90°, display as upper semicircle
            self.ax.set_thetamin(0)
            self.ax.set_thetamax(90)
        else:
            # 0-180: full semicircle (z-axis at top, equator at 90°, -z at bottom)
            self.ax.set_thetamin(0)
            self.ax.set_thetamax(180)

        self.ax.grid(True, alpha=0.30, linestyle="--", linewidth=0.6)
        subtitle = ("θ: 0–90° (folded)" if proj == "0-90 folded"
                    else "θ: 0–180°  (N=z-axis, S=−z)")
        self.ax.set_title(
            f"Metal-centred polar comparison  [{subtitle}]\n"
            r"$\theta$ = angle from z-axis   ·   r = distance (Å)",
            fontsize=9, pad=12,
        )

        if legend_handles:
            self.ax.legend(
                handles=legend_handles,
                loc="lower right",
                bbox_to_anchor=(1.0, -0.08),
                fontsize=8, frameon=True, framealpha=0.92,
                borderpad=0.5, labelspacing=0.3,
            )

        # ── tooltip (C1) ──
        self._attach_tooltips()

        self.canvas.draw_idle()

        # update analysis tab
        self._refresh_analysis()

    # ══════════════════════════════════════════════════════════════════════════
    # Tooltip (C1)
    # ══════════════════════════════════════════════════════════════════════════

    def _attach_tooltips(self):
        """Attach hover tooltips using mplcursors if available, else motion event."""
        if _MPLCURSORS:
            self._attach_mplcursors()
        else:
            self._attach_motion_tooltip()

    def _attach_mplcursors(self):
        import mplcursors  # noqa: F811
        if not self.structures:
            return

        for sc, sidx, atom_indices, scat_th, scat_r in self._scatter_meta:
            st = self.structures[sidx]
            st_metal_pos = self._get_metal_position(st)  # per-structure origin
            if st_metal_pos is None:
                continue
            cursor = mplcursors.cursor(sc, hover=True)

            @cursor.connect("add")
            def _(sel, _st=st, _indices=atom_indices, _mpos=st_metal_pos):
                idx_local = sel.index
                if idx_local >= len(_indices):
                    return
                atom_idx = _indices[idx_local]
                el = _st.elements[atom_idx]
                dist = float(np.linalg.norm(_st.coords[atom_idx] - _mpos))
                sel.annotation.set_text(
                    f"{atom_idx+1}  {el}\n"
                    f"d(M) = {dist:.3f} Å\n"
                    f"struct: {_st.label}"
                )
                sel.annotation.get_bbox_patch().set(
                    facecolor="#ffffee", alpha=0.95, edgecolor="#999999"
                )

    def _attach_motion_tooltip(self):
        """Fallback tooltip via motion_notify_event."""
        if hasattr(self, "_tooltip_ann"):
            try:
                self._tooltip_ann.remove()
            except Exception:
                pass
        self._tooltip_ann = None

        if not self.structures:
            return

        def _on_motion(event):
            if event.inaxes != self.ax:
                if self._tooltip_ann:
                    self._tooltip_ann.set_visible(False)
                    self.canvas.draw_idle()
                return

            hit = False
            for sc, sidx, atom_indices, scat_th, scat_r in self._scatter_meta:
                st = self.structures[sidx]
                st_metal_pos = self._get_metal_position(st)  # per-structure origin
                for k, (th, r) in enumerate(zip(scat_th, scat_r)):
                    dth = event.xdata - th
                    dr  = event.ydata - r
                    if dth * dth + dr * dr < 0.005:
                        atom_idx = atom_indices[k]
                        el   = st.elements[atom_idx]
                        dist = float(np.linalg.norm(st.coords[atom_idx] - st_metal_pos))
                        txt  = f"{atom_idx+1}  {el}  d(M)={dist:.3f} Å"
                        if self._tooltip_ann is None:
                            self._tooltip_ann = self.ax.annotate(
                                txt, xy=(th, r), xytext=(10, 10),
                                textcoords="offset points",
                                fontsize=8,
                                bbox=dict(boxstyle="round,pad=0.3",
                                          fc="#ffffee", ec="#999999", alpha=0.95),
                                zorder=20,
                            )
                        else:
                            self._tooltip_ann.set_text(txt)
                            self._tooltip_ann.xy = (th, r)
                            self._tooltip_ann.set_visible(True)
                        self.canvas.draw_idle()
                        hit = True
                        break
                if hit:
                    break

            if not hit and self._tooltip_ann:
                self._tooltip_ann.set_visible(False)
                self.canvas.draw_idle()

        self.fig.canvas.mpl_connect("motion_notify_event", _on_motion)

    # ══════════════════════════════════════════════════════════════════════════
    # Context menu (C2)
    # ══════════════════════════════════════════════════════════════════════════

    def _show_context_menu(self, event):
        try:
            self._ctx_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self._ctx_menu.grab_release()

    # ══════════════════════════════════════════════════════════════════════════
    # Analysis tab (A3)
    # ══════════════════════════════════════════════════════════════════════════

    # ─────────────────────────────────────────────────────────────────────────
    # Analysis helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _write_tab(self, widget: tk.Text, text: str):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", text)
        widget.configure(state="disabled")

    def _refresh_analysis(self):
        if not self.structures:
            for w in [self._txt_rmsd, self._txt_ml, self._txt_geom,
                      self._txt_trans, self._txt_polar, self._txt_shannon]:
                self._write_tab(w, "Load XYZ files to see analysis.")
            return

        ref  = self.structures[self.reference_index]
        W    = 72   # column width

        # ── collect per-structure bonded lists and metal positions ──
        struct_data = []
        for st in self.structures:
            mp     = self._get_metal_position(st)
            midx   = getattr(st, "metal_index", 0)
            bonded = _get_bonded(st.elements, st.coords, midx, mp) if mp is not None else []
            vecs   = {i: st.coords[i] - mp for i in bonded} if mp is not None else {}
            dists  = {i: float(np.linalg.norm(v)) for i, v in vecs.items()}
            angles = []
            blist  = list(bonded)
            for a in range(len(blist)):
                for b in range(a+1, len(blist)):
                    angles.append(_lml_angle(vecs[blist[a]], vecs[blist[b]]))
            struct_data.append(dict(st=st, mp=mp, midx=midx,
                                    bonded=bonded, vecs=vecs,
                                    dists=dists, angles=angles))

        ref_d = struct_data[self.reference_index]

        # ════════════════════════════════════════════════════════════════════
        # Tab 1 – RMSD
        # ════════════════════════════════════════════════════════════════════
        L = ["═"*W, "  RMSD vs reference", "─"*W,
             f"  {'Structure':<30}  {'Raw RMSD (Å)':>14}  {'Kabsch RMSD (Å)':>16}",
             "─"*W]
        for sd in struct_data:
            st = sd["st"]
            if st is ref:
                L.append(f"  {st.label:<30}  {'(reference)':>14}")
                continue
            if len(st.coords) != len(ref.coords):
                L.append(f"  {st.label:<30}  atom count mismatch")
                continue
            raw = float(np.sqrt(np.mean(np.sum((st.coords - ref.coords)**2, axis=1))))
            kab, _ = kabsch_rmsd(st.coords, ref.coords)
            L.append(f"  {st.label:<30}  {raw:>14.4f}  {kab:>16.4f}")
        L += ["═"*W]
        self._write_tab(self._txt_rmsd, "\n".join(L))

        # ════════════════════════════════════════════════════════════════════
        # Tab 2 – M–L distance table with Δ columns
        # ════════════════════════════════════════════════════════════════════
        L = ["═"*W, "  Metal–Ligand Distances and Δ vs Reference", "─"*W]

        # header: Atom | El | ref_dist | comp1_dist | Δ1 | comp2_dist | Δ2 ...
        comps = [sd for sd in struct_data if sd["st"] is not ref]
        hdr = f"  {'#':>4}  {'El':<3}  {'Ref (Å)':>9}"
        for sd in comps:
            lbl = sd["st"].label[:10]
            hdr += f"  {lbl:>10}  {'Δ (Å)':>7}"
        L.append(hdr)
        L.append("─"*W)

        ref_bonded = ref_d["bonded"]
        ref_mp     = ref_d["mp"]
        ref_midx   = ref_d["midx"]

        # mean row accumulator
        ref_mean_d = []
        comp_mean_d = [[] for _ in comps]

        for i in ref_bonded:
            el      = ref.elements[i]
            ref_dist = ref_d["dists"].get(i, float("nan"))
            ref_mean_d.append(ref_dist)
            row = f"  {i+1:>4}  {el:<3}  {ref_dist:>9.4f}"
            for ci, sd in enumerate(comps):
                st = sd["st"]
                # match by atom index (isostructural assumption)
                if i < len(st.coords):
                    mp2    = sd["mp"]
                    dist2  = float(np.linalg.norm(st.coords[i] - mp2)) if mp2 is not None else float("nan")
                    delta  = dist2 - ref_dist
                    comp_mean_d[ci].append(dist2)
                    row += f"  {dist2:>10.4f}  {delta:>+7.4f}"
                else:
                    row += f"  {'—':>10}  {'—':>7}"
            L.append(row)

        # mean row
        L.append("─"*W)
        mean_row = f"  {'mean':>4}  {'':3}  {float(np.mean(ref_mean_d)) if ref_mean_d else float('nan'):>9.4f}"
        for ci, sd in enumerate(comps):
            if comp_mean_d[ci] and ref_mean_d:
                m  = float(np.mean(comp_mean_d[ci]))
                dm = m - float(np.mean(ref_mean_d))
                mean_row += f"  {m:>10.4f}  {dm:>+7.4f}"
            else:
                mean_row += f"  {'—':>10}  {'—':>7}"
        L.append(mean_row)
        L += ["═"*W]
        self._write_tab(self._txt_ml, "\n".join(L))

        # ════════════════════════════════════════════════════════════════════
        # Tab 3 – Coordination geometry indices
        # ════════════════════════════════════════════════════════════════════
        L = ["═"*W, "  Coordination Geometry Indices", "─"*W]

        for sd in struct_data:
            st     = sd["st"]
            bonded = sd["bonded"]
            angles = sd["angles"]
            cn     = len(bonded)
            tag    = " [ref]" if st is ref else ""
            m_el   = st.elements[sd["midx"]] if sd["midx"] < len(st.elements) else "?"

            L.append(f"  {st.label}{tag}   ({m_el}, CN={cn})")
            L.append("─"*W)

            if not angles:
                L.append("  (no coordination sphere detected)")
                L.append("")
                continue

            # τ₄ / τ₅
            tau4 = calc_tau4(angles) if cn == 4 else None
            tau5 = calc_tau5(angles) if cn == 5 else None
            sigma = calc_angular_deviation(angles, cn)

            if tau4 is not None:
                geom = "square planar" if tau4 < 0.25 else ("tetrahedral" if tau4 > 0.75 else "intermediate")
                L.append(f"    τ₄ = {tau4:.3f}   ({geom})")
            if tau5 is not None:
                geom = "square pyramidal" if tau5 < 0.25 else ("trig. bipyramidal" if tau5 > 0.75 else "intermediate")
                L.append(f"    τ₅ = {tau5:.3f}   ({geom})")
            if sigma is not None:
                L.append(f"    Σ  = {sigma:.2f}°  (angular deviation from ideal)")
            else:
                L.append(f"    Σ  = n/a  (no ideal reference for CN={cn})")

            # All L-M-L angles
            blist = list(bonded)
            L.append(f"    L-M-L angles ({len(angles)} pairs):")
            row = "      "
            for a_idx in range(len(blist)):
                for b_idx in range(a_idx+1, len(blist)):
                    i, j = blist[a_idx], blist[b_idx]
                    ang  = _lml_angle(sd["vecs"][i], sd["vecs"][j])
                    el_i = st.elements[i]; el_j = st.elements[j]
                    entry = f"{i+1}{el_i}-{j+1}{el_j}: {ang:.1f}°"
                    if len(row)+len(entry) > 70:
                        L.append(row)
                        row = "      "
                    row += entry + "  "
            if row.strip():
                L.append(row)
            L.append("")

        L += ["═"*W]
        self._write_tab(self._txt_geom, "\n".join(L))

        # ════════════════════════════════════════════════════════════════════
        # Tab 4 – Trans influence
        # ════════════════════════════════════════════════════════════════════
        L = ["═"*W,
             "  Trans-influence Indicator  (L-M-L ≥ 150°)",
             "  Larger |Δd| = stronger trans influence asymmetry",
             "─"*W]

        for sd in struct_data:
            st     = sd["st"]
            bonded = sd["bonded"]
            tag    = " [ref]" if st is ref else ""
            m_el   = st.elements[sd["midx"]] if sd["midx"] < len(st.elements) else "?"
            trans  = calc_sigma_trans(st.elements, st.coords,
                                      sd["midx"], sd["mp"], bonded) if sd["mp"] is not None else []
            L.append(f"  {st.label}{tag}  ({m_el})")
            if not trans:
                L.append("    (no trans pairs found, L-M-L < 150°)")
            else:
                L.append(f"    {'Pair':<20}  {'Angle (°)':>10}  {'d₁ (Å)':>8}  {'d₂ (Å)':>8}  {'|Δd| (Å)':>10}")
                L.append("    " + "─"*58)
                for (i, j, ang, delta_d) in sorted(trans, key=lambda x: -x[2]):
                    el_i = st.elements[i]; el_j = st.elements[j]
                    d_i  = sd["dists"][i]; d_j  = sd["dists"][j]
                    pair = f"{i+1}{el_i}–{j+1}{el_j}"
                    L.append(f"    {pair:<20}  {ang:>10.2f}  {d_i:>8.4f}  {d_j:>8.4f}  {delta_d:>10.4f}")
            L.append("")

        L += ["═"*W]
        self._write_tab(self._txt_trans, "\n".join(L))

        # ════════════════════════════════════════════════════════════════════
        # Tab 5 – Polar plot statistics (θ / r)
        # ════════════════════════════════════════════════════════════════════
        proj = self.proj_var.get()
        L = ["═"*W,
             f"  Polar Statistics  (projection: {proj})",
             "─"*W,
             f"  {'Structure':<24}  {'r̄ (Å)':>7}  {'σr':>6}  {'θ̄ (°)':>7}  {'σθ':>6}  {'CN':>4}"]
        L.append("─"*W)

        for sd in struct_data:
            st     = sd["st"]
            bonded = sd["bonded"]
            tag    = " [ref]" if st is ref else ""
            mp     = sd["mp"]
            if mp is None or not bonded:
                L.append(f"  {st.label+tag:<24}  (no metal position)")
                continue
            ps = polar_stats(st.elements, st.coords, mp, bonded, proj)
            L.append(
                f"  {st.label+tag:<24}"
                f"  {ps.get('r_mean',float('nan')):>7.4f}"
                f"  {ps.get('r_std', float('nan')):>6.4f}"
                f"  {ps.get('theta_mean',float('nan')):>7.2f}"
                f"  {ps.get('theta_std', float('nan')):>6.2f}"
                f"  {len(bonded):>4}"
            )

        L += ["", "─"*W, "  Per-atom polar coordinates (coord sphere only)", "─"*W]
        for sd in struct_data:
            st     = sd["st"]
            bonded = sd["bonded"]
            mp     = sd["mp"]
            tag    = " [ref]" if st is ref else ""
            if mp is None or not bonded:
                continue
            L.append(f"  {st.label}{tag}")
            L.append(f"    {'#':>4}  {'El':<3}  {'r (Å)':>8}  {'θ (°)':>8}  {'φ (°)':>8}")
            for i in bonded:
                rel   = st.coords[i] - mp
                r     = float(np.linalg.norm(rel))
                theta = float(np.degrees(np.arccos(np.clip(rel[2]/(r+1e-15),-1,1))))
                phi   = float(np.degrees(np.arctan2(rel[1], rel[0])))
                L.append(f"    {i+1:>4}  {st.elements[i]:<3}  {r:>8.4f}  {theta:>8.2f}  {phi:>8.2f}")
            L.append("")

        L += ["═"*W]
        self._write_tab(self._txt_polar, "\n".join(L))

        # ════════════════════════════════════════════════════════════════════
        # Tab 6 – Ionic radii correlation
        # ════════════════════════════════════════════════════════════════════
        L = ["═"*W,
             "  Ionic Radii Correlation  (Shannon 6-coord radii)",
             "─"*W,
             f"  {'Structure':<24}  {'Metal':<4}  {'r_ion (Å)':>10}  {'M–L mean (Å)':>14}  {'M–L range (Å)':>15}"]
        L.append("─"*W)

        series = []
        for sd in struct_data:
            st   = sd["st"]
            midx = sd["midx"]
            tag  = " [ref]" if st is ref else ""
            m_el = st.elements[midx] if midx < len(st.elements) else "?"
            r_ion = SHANNON_RADII.get(m_el)
            dists_vals = list(sd["dists"].values())
            ml_mean = float(np.mean(dists_vals)) if dists_vals else float("nan")
            ml_range = (float(np.max(dists_vals) - np.min(dists_vals))
                        if len(dists_vals) > 1 else 0.0)
            r_ion_str = f"{r_ion:.4f}" if r_ion else "N/A"
            L.append(
                f"  {st.label+tag:<24}  {m_el:<4}  {r_ion_str:>10}"
                f"  {ml_mean:>14.4f}  {ml_range:>15.4f}"
            )
            if r_ion and not np.isnan(ml_mean):
                series.append((m_el, r_ion, ml_mean))

        if len(series) >= 2:
            r_ions = np.array([s[1] for s in series])
            ml_means = np.array([s[2] for s in series])
            # linear regression
            A = np.vstack([r_ions, np.ones(len(r_ions))]).T
            slope, intercept = np.linalg.lstsq(A, ml_means, rcond=None)[0]
            ss_res = float(np.sum((ml_means - (slope*r_ions+intercept))**2))
            ss_tot = float(np.sum((ml_means - ml_means.mean())**2))
            r2     = 1 - ss_res/ss_tot if ss_tot > 1e-12 else float("nan")
            L += ["─"*W,
                  f"  Linear fit:  M–L(Å) = {slope:.4f} × r_ion + {intercept:.4f}",
                  f"  R² = {r2:.4f}",
                  "  (Ideal slope ≈ 1.0 for purely ionic M–L bonds)"]

        L += ["═"*W]
        self._write_tab(self._txt_shannon, "\n".join(L))

        # status
        self._set_status(
            f"{len(self.structures)} structure(s)  ·  "
            f"ref: {ref.label}"
        )

    def _set_analysis(self, text: str):
        # legacy helper: writes to RMSD tab
        self._write_tab(self._txt_rmsd, text)

    # ══════════════════════════════════════════════════════════════════════════
    # Analysis plots (popup windows)
    # ══════════════════════════════════════════════════════════════════════════

    def _analysis_struct_data(self):
        """Collect per-structure bonded/dist/angle data (shared by plot methods)."""
        data = []
        for st in self.structures:
            mp    = self._get_metal_position(st)
            midx  = getattr(st, "metal_index", 0)
            bonded = (_get_bonded(st.elements, st.coords, midx, mp)
                      if mp is not None else [])
            vecs  = {i: st.coords[i] - mp for i in bonded} if mp is not None else {}
            dists = {i: float(np.linalg.norm(v)) for i, v in vecs.items()}
            data.append(dict(st=st, mp=mp, midx=midx,
                             bonded=bonded, vecs=vecs, dists=dists))
        return data

    def _make_plot_window(self, title: str, figsize=(8, 5)):
        """Create a Toplevel with an embedded matplotlib figure. Returns (win, fig, ax)."""
        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry(f"{figsize[0]*100}x{figsize[1]*100+60}")
        fig = plt.Figure(figsize=figsize, dpi=100, facecolor="#F7F8FA")
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(canvas, win).update()
        return win, fig, canvas

    def _plot_ml_trends(self):
        """
        Plot M-L distance per ligand atom across all structures.
        One line per coordination-sphere atom (matched by index).
        X axis = structure label, Y axis = M-L distance (Å).
        """
        if not self.structures:
            return
        data = self._analysis_struct_data()
        ref_d = data[self.reference_index]
        ref_bonded = ref_d["bonded"]
        if not ref_bonded:
            messagebox.showinfo("M-L plot", "No coordination sphere detected.")
            return

        win, fig, canvas = self._make_plot_window("M–L Distance Trends", (9, 5))
        ax = fig.add_subplot(1, 1, 1)

        x_labels = [sd["st"].label for sd in data]
        x_pos    = list(range(len(data)))

        # color per ligand atom
        cmap = matplotlib.colormaps["tab10"]
        for k, atom_i in enumerate(ref_bonded):
            el   = self.structures[self.reference_index].elements[atom_i]
            y    = []
            for sd in data:
                if atom_i in sd["dists"]:
                    y.append(sd["dists"][atom_i])
                elif atom_i < len(sd["st"].coords) and sd["mp"] is not None:
                    y.append(float(np.linalg.norm(
                        sd["st"].coords[atom_i] - sd["mp"])))
                else:
                    y.append(float("nan"))
            color = cmap(k % cmap.N)
            ax.plot(x_pos, y, "o-", color=color,
                    label=f"#{atom_i+1} {el}", linewidth=1.5, markersize=6)

        # mean line
        means = []
        for sd in data:
            dv = list(sd["dists"].values())
            means.append(float(np.mean(dv)) if dv else float("nan"))
        ax.plot(x_pos, means, "k--", linewidth=2, label="mean", zorder=10)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("M–L distance (Å)")
        ax.set_title("Metal–Ligand Distance Trends across Isostructural Series")
        ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
        ax.grid(True, alpha=0.3, linestyle="--")
        fig.tight_layout()
        canvas.draw()

    def _plot_ionic_radii(self):
        """
        Scatter plot: Shannon ionic radius vs mean M-L distance.
        Includes linear regression line and R².
        """
        if not self.structures:
            return
        data = self._analysis_struct_data()

        series = []
        for sd in data:
            midx  = sd["midx"]
            m_el  = sd["st"].elements[midx] if midx < len(sd["st"].elements) else "?"
            r_ion = SHANNON_RADII.get(m_el)
            dv    = list(sd["dists"].values())
            if r_ion is None or not dv:
                continue
            ml_mean = float(np.mean(dv))
            series.append((sd["st"].label, m_el, r_ion, ml_mean))

        if len(series) < 2:
            messagebox.showinfo("Ionic radii plot",
                                "Need at least 2 structures with known Shannon radii.")
            return

        win, fig, canvas = self._make_plot_window("Ionic Radii vs M–L Distance", (7, 5))
        ax = fig.add_subplot(1, 1, 1)

        labels  = [s[0] for s in series]
        m_els   = [s[1] for s in series]
        r_ions  = np.array([s[2] for s in series])
        ml_means = np.array([s[3] for s in series])

        cmap = matplotlib.colormaps["tab10"]
        for k, (lbl, m_el, ri, ml) in enumerate(series):
            ax.scatter(ri, ml, s=80, color=cmap(k % cmap.N),
                       zorder=5, label=f"{lbl} ({m_el})")
            ax.annotate(m_el, (ri, ml), textcoords="offset points",
                        xytext=(5, 4), fontsize=8)

        # linear regression
        A = np.vstack([r_ions, np.ones(len(r_ions))]).T
        slope, intercept = np.linalg.lstsq(A, ml_means, rcond=None)[0]
        x_fit = np.linspace(r_ions.min() - 0.05, r_ions.max() + 0.05, 100)
        y_fit = slope * x_fit + intercept
        ss_res = float(np.sum((ml_means - (slope*r_ions+intercept))**2))
        ss_tot = float(np.sum((ml_means - ml_means.mean())**2))
        r2     = 1 - ss_res/ss_tot if ss_tot > 1e-12 else float("nan")
        ax.plot(x_fit, y_fit, "r--", linewidth=1.5,
                label=f"fit: y={slope:.3f}x+{intercept:.3f}  R²={r2:.3f}")

        ax.set_xlabel("Shannon ionic radius (Å, 6-coord)")
        ax.set_ylabel("Mean M–L distance (Å)")
        ax.set_title("Shannon Ionic Radius vs Mean M–L Distance")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, linestyle="--")
        fig.tight_layout()
        canvas.draw()

    def _plot_polar_dist(self):
        """
        Histogram + KDE of r and θ distributions for coordination sphere atoms,
        one colour per structure — allows visual comparison of spread.
        """
        if not self.structures:
            return
        data = self._analysis_struct_data()
        proj = self.proj_var.get()

        win, fig, canvas = self._make_plot_window("Polar Distribution", (10, 4))
        ax_r   = fig.add_subplot(1, 2, 1)
        ax_th  = fig.add_subplot(1, 2, 2)

        cmap = matplotlib.colormaps["tab10"]
        ref  = self.structures[self.reference_index]

        for k, sd in enumerate(data):
            st     = sd["st"]
            bonded = sd["bonded"]
            mp     = sd["mp"]
            if not bonded or mp is None:
                continue
            rel   = st.coords[bonded] - mp
            r     = np.linalg.norm(rel, axis=1)
            theta = np.arccos(np.clip(rel[:, 2] / (r + 1e-15), -1, 1))
            if proj == "0-90 folded":
                theta = np.minimum(theta, np.pi - theta)
            theta_deg = np.degrees(theta)
            color = cmap(k % cmap.N)
            tag   = " [ref]" if st is ref else ""
            lbl   = st.label + tag

            # r histogram
            ax_r.hist(r, bins=min(8, len(r)), alpha=0.55, color=color,
                      label=lbl, density=True)
            ax_r.axvline(float(np.mean(r)), color=color,
                         linestyle="--", linewidth=1.2)

            # θ histogram
            ax_th.hist(theta_deg, bins=min(8, len(theta_deg)),
                       alpha=0.55, color=color, label=lbl, density=True)
            ax_th.axvline(float(np.mean(theta_deg)), color=color,
                          linestyle="--", linewidth=1.2)

        ax_r.set_xlabel("r (Å)")
        ax_r.set_ylabel("Density")
        ax_r.set_title("M–L Distance Distribution")
        ax_r.legend(fontsize=8)
        ax_r.grid(True, alpha=0.3, linestyle="--")

        ax_th.set_xlabel("θ (°)")
        ax_th.set_ylabel("Density")
        ax_th.set_title(f"Polar Angle Distribution ({proj})")
        ax_th.legend(fontsize=8)
        ax_th.grid(True, alpha=0.3, linestyle="--")

        fig.tight_layout()
        canvas.draw()

    # ══════════════════════════════════════════════════════════════════════════
    # Analysis CSV export
    # ══════════════════════════════════════════════════════════════════════════

    def _export_analysis_csv(self):
        """Export all numeric analysis data as a multi-sheet CSV (one block per section)."""
        if not self.structures:
            messagebox.showinfo("Export CSV", "No structures loaded.")
            return
        path = filedialog.asksaveasfilename(
            title="Export analysis CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return

        data = self._analysis_struct_data()
        ref  = self.structures[self.reference_index]
        ref_d = data[self.reference_index]
        proj  = self.proj_var.get()
        rows: List[str] = []

        def sec(title):
            rows.append("")
            rows.append(f"# {title}")

        # ── RMSD ──
        sec("RMSD")
        rows.append("structure,raw_rmsd_A,kabsch_rmsd_A")
        for sd in data:
            st = sd["st"]
            if st is ref:
                rows.append(f"{st.label},reference,reference")
                continue
            if len(st.coords) != len(ref.coords):
                rows.append(f"{st.label},atom_mismatch,atom_mismatch")
                continue
            raw = float(np.sqrt(np.mean(np.sum((st.coords - ref.coords)**2, axis=1))))
            kab, _ = kabsch_rmsd(st.coords, ref.coords)
            rows.append(f"{st.label},{raw:.6f},{kab:.6f}")

        # ── M-L distances ──
        sec("M-L distances")
        comp_labels = [sd["st"].label for sd in data]
        rows.append("atom_index,element," + ",".join(comp_labels))
        all_atoms = sorted(set(i for sd in data for i in sd["bonded"]))
        for i in all_atoms:
            el  = ref.elements[i] if i < len(ref.elements) else "?"
            dv  = []
            for sd in data:
                d = sd["dists"].get(i)
                if d is None and i < len(sd["st"].coords) and sd["mp"] is not None:
                    d = float(np.linalg.norm(sd["st"].coords[i] - sd["mp"]))
                dv.append(f"{d:.6f}" if d is not None else "")
            rows.append(f"{i+1},{el}," + ",".join(dv))

        # ── Geometry ──
        sec("Geometry indices")
        rows.append("structure,metal,CN,tau4,tau5,sigma_deg")
        for sd in data:
            st     = sd["st"]
            bonded = sd["bonded"]
            angles_all = []
            blist  = list(bonded)
            for a in range(len(blist)):
                for b in range(a+1, len(blist)):
                    angles_all.append(_lml_angle(sd["vecs"][blist[a]], sd["vecs"][blist[b]]))
            cn    = len(bonded)
            m_el  = st.elements[sd["midx"]] if sd["midx"] < len(st.elements) else "?"
            tau4  = calc_tau4(angles_all) if cn == 4 else ""
            tau5  = calc_tau5(angles_all) if cn == 5 else ""
            sigma = calc_angular_deviation(angles_all, cn)
            rows.append(
                f"{st.label},{m_el},{cn},"
                f"{tau4 if tau4 is not None and tau4 != '' else ''},"
                f"{tau5 if tau5 is not None and tau5 != '' else ''},"
                f"{sigma:.4f}" if sigma is not None else
                f"{st.label},{m_el},{cn},{tau4 or ''},{tau5 or ''},")

        # ── Polar stats ──
        sec("Polar statistics")
        rows.append("structure,r_mean_A,r_std_A,r_min_A,r_max_A,theta_mean_deg,theta_std_deg,CN")
        for sd in data:
            st = sd["st"]
            mp = sd["mp"]
            if mp is None or not sd["bonded"]:
                continue
            ps = polar_stats(st.elements, st.coords, mp, sd["bonded"], proj)
            rows.append(
                f"{st.label},"
                f"{ps.get('r_mean',float('nan')):.6f},"
                f"{ps.get('r_std', float('nan')):.6f},"
                f"{ps.get('r_min', float('nan')):.6f},"
                f"{ps.get('r_max', float('nan')):.6f},"
                f"{ps.get('theta_mean',float('nan')):.4f},"
                f"{ps.get('theta_std', float('nan')):.4f},"
                f"{len(sd['bonded'])}"
            )

        # ── Shannon radii ──
        sec("Ionic radii correlation")
        rows.append("structure,metal,shannon_r_ion_A,ml_mean_A,ml_range_A")
        for sd in data:
            st   = sd["st"]
            midx = sd["midx"]
            m_el = st.elements[midx] if midx < len(st.elements) else "?"
            r_ion = SHANNON_RADII.get(m_el, "")
            dv    = list(sd["dists"].values())
            ml_mean  = float(np.mean(dv)) if dv else ""
            ml_range = float(np.max(dv) - np.min(dv)) if len(dv) > 1 else 0.0
            rows.append(
                f"{st.label},{m_el},{r_ion},"
                f"{ml_mean:.6f}" if isinstance(ml_mean, float) else
                f"{st.label},{m_el},{r_ion},,")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")
        self._set_status(f"Exported analysis CSV: {os.path.basename(path)}")

    def _export_analysis_report(self):
        """Export full analysis as a plain-text report file."""
        if not self.structures:
            messagebox.showinfo("Export report", "No structures loaded.")
            return
        path = filedialog.asksaveasfilename(
            title="Export analysis report",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return

        tabs = [
            ("RMSD",         self._txt_rmsd),
            ("M-L Delta",    self._txt_ml),
            ("Geometry",     self._txt_geom),
            ("Trans",        self._txt_trans),
            ("Polar stats",  self._txt_polar),
            ("Ionic radii",  self._txt_shannon),
        ]
        sep = "\n" + "═"*72 + "\n"
        parts = []
        for title, widget in tabs:
            content = widget.get("1.0", "end").strip()
            parts.append(f"{'═'*72}\n  {title}\n{'═'*72}\n{content}")

        with open(path, "w", encoding="utf-8") as f:
            f.write(sep.join(parts) + "\n")
        self._set_status(f"Exported report: {os.path.basename(path)}")

    # ══════════════════════════════════════════════════════════════════════════
    # Export (D4)
    # ══════════════════════════════════════════════════════════════════════════

    def _export(self, fmt: str):
        ext = fmt.lower()
        path = filedialog.asksaveasfilename(
            title=f"Save figure as {ext.upper()}",
            defaultextension=f".{ext}",
            filetypes=[(ext.upper(), f"*.{ext}"), ("All files", "*.*")],
        )
        if not path:
            return
        self.fig.savefig(path, dpi=300, bbox_inches="tight")
        self._set_status(f"Saved {ext.upper()}: {os.path.basename(path)}")

    def _export_csv(self):
        if not self.structures:
            messagebox.showinfo("Export CSV", "No structures loaded.")
            return
        path = filedialog.asksaveasfilename(
            title="Export numeric data as CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return

        metal_pos = self._get_metal_position()
        rows = ["structure,atom_index,element,x,y,z,r_from_metal,theta_deg"]
        for st in self.structures:
            theta, r = self._polar_coords(st.coords, metal_pos)
            for i, (el, coord) in enumerate(zip(st.elements, st.coords)):
                rows.append(
                    f"{st.label},{i+1},{el},"
                    f"{coord[0]:.6f},{coord[1]:.6f},{coord[2]:.6f},"
                    f"{r[i]:.6f},{float(np.degrees(theta[i])):.4f}"
                )
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(rows))
        self._set_status(f"Exported CSV: {os.path.basename(path)}")

    # ══════════════════════════════════════════════════════════════════════════
    # Status bar (D3)
    # ══════════════════════════════════════════════════════════════════════════

    def _set_status(self, msg: str):
        self.status_var.set(f"  {msg}")

    # ══════════════════════════════════════════════════════════════════════════
    # Session save / restore (E1)
    # ══════════════════════════════════════════════════════════════════════════

    def _current_display_options(self) -> dict:
        """Collect display options that should be stored in config and sessions."""
        return {
            "show_h":            self.show_h_var.get(),
            "show_bonds":        self.show_bonds_var.get(),
            "show_atoms":        self.show_atoms_var.get(),
            "show_labels":       self.show_labels_var.get(),
            "show_displacement": self.show_displacement_var.get(),
            "use_kabsch":        self.use_kabsch_var.get(),
            "proj":              self.proj_var.get(),
            "bond_scale":        self.bond_scale_var.get(),
            "bond_width":        self.bond_width_var.get(),
            "bond_alpha":        self.bond_alpha_var.get(),
            "atom_size":         self.atom_size_var.get(),
        }

    def _apply_display_options(self, opts: dict):
        """Apply display options from a saved session or config dictionary."""
        if not isinstance(opts, dict):
            return
        bool_targets = {
            "show_h": self.show_h_var,
            "show_bonds": self.show_bonds_var,
            "show_atoms": self.show_atoms_var,
            "show_labels": self.show_labels_var,
            "show_displacement": self.show_displacement_var,
            "use_kabsch": self.use_kabsch_var,
        }
        for key, var in bool_targets.items():
            if key in opts:
                var.set(bool(opts[key]))

        str_targets = {
            "proj": self.proj_var,
            "bond_scale": self.bond_scale_var,
            "bond_width": self.bond_width_var,
            "bond_alpha": self.bond_alpha_var,
            "atom_size": self.atom_size_var,
        }
        for key, var in str_targets.items():
            if key in opts:
                var.set(str(opts[key]))

    def _structure_to_session_dict(self, st: XYZStructure) -> dict:
        axis = getattr(st, "align_axis", None)
        return {
            "label": st.label,
            "path": st.path,
            "color": st.color,
            "memo": st.memo,
            "visible": getattr(st, "visible", True),
            "metal_index": int(getattr(st, "metal_index", 0)),
            "use_manual": bool(getattr(st, "use_manual", False)),
            "mx": float(getattr(st, "mx", 0.0)),
            "my": float(getattr(st, "my", 0.0)),
            "mz": float(getattr(st, "mz", 0.0)),
            "use_align": bool(getattr(st, "use_align", False)),
            "align_axis": axis.tolist() if axis is not None else None,
            "align_Cn": int(getattr(st, "align_Cn", 0)),
            "coord_sphere": list(getattr(st, "coord_sphere", None) or []),
        }

    def _structure_from_session_dict(self, item: dict) -> XYZStructure:
        path = item.get("path", "")
        if not path or not os.path.exists(path):
            raise FileNotFoundError(path or "<empty path>")
        elements, coords = load_xyz(path)
        label = item.get("label") or os.path.splitext(os.path.basename(path))[0]

        metal_index = int(item.get("metal_index", 0))
        if not (0 <= metal_index < len(elements)):
            metal_index = 0

        st = XYZStructure(
            label=label,
            path=path,
            elements=elements,
            coords=coords,
            color=item.get("color", "#1a8c5e"),
            memo=item.get("memo", ""),
            visible=bool(item.get("visible", True)),
            metal_index=metal_index,
            use_manual=bool(item.get("use_manual", False)),
            mx=float(item.get("mx", 0.0)),
            my=float(item.get("my", 0.0)),
            mz=float(item.get("mz", 0.0)),
            use_align=bool(item.get("use_align", False)),
            align_Cn=int(item.get("align_Cn", 0)),
        )

        axis = item.get("align_axis")
        if isinstance(axis, list) and len(axis) == 3:
            try:
                st.align_axis = np.asarray(axis, dtype=float)
            except Exception:
                st.align_axis = None

        sphere = item.get("coord_sphere")
        if isinstance(sphere, list):
            st.coord_sphere = [int(i) for i in sphere if isinstance(i, int) or str(i).isdigit()]
        else:
            st.coord_sphere = None
        return st

    def _collect_session(self) -> dict:
        """Return a complete serialisable viewer-session dictionary."""
        # Pull current checkbox states into the structure objects before saving.
        for idx, var in enumerate(self._structure_vars):
            if idx < len(self.structures):
                self.structures[idx].visible = var.get()

        return {
            "format": "CoordinationPolarViewerSession",
            "format_version": 1,
            "app_id": PLUGIN_INFO["id"],
            "app_version": PLUGIN_INFO["version"],
            "reference_index": int(self.reference_index),
            "display_options": self._current_display_options(),
            "structures": [self._structure_to_session_dict(st) for st in self.structures],
        }

    def save_session_dialog(self):
        if not self.structures:
            messagebox.showinfo("Save session", "No structures loaded.")
            return
        path = filedialog.asksaveasfilename(
            title="Save Coordination Polar Viewer session",
            defaultextension=".cpvsession",
            filetypes=SESSION_FILETYPES,
        )
        if not path:
            return
        self.save_session(path)

    def save_session(self, path: str):
        session = self._collect_session()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(session, f, indent=2)
        self._cfg["last_session_path"] = path
        save_config({**self._collect_config(), "last_session_path": path})
        self._set_status(f"Saved session: {os.path.basename(path)}")

    def load_session_dialog(self):
        initial = self._cfg.get("last_session_path", "")
        kwargs = {}
        if initial:
            kwargs["initialdir"] = os.path.dirname(initial)
        path = filedialog.askopenfilename(
            title="Load Coordination Polar Viewer session",
            filetypes=SESSION_FILETYPES,
            **kwargs,
        )
        if not path:
            return
        self.load_session(path)

    def load_session(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            session = json.load(f)

        structures_data = session.get("structures", [])
        if not isinstance(structures_data, list):
            raise ValueError("Invalid session file: 'structures' must be a list.")

        loaded: List[XYZStructure] = []
        missing: List[str] = []
        for item in structures_data:
            try:
                loaded.append(self._structure_from_session_dict(item))
            except Exception:
                missing.append(str(item.get("path", "<unknown>")))

        if not loaded:
            msg = "No structures could be loaded from this session."
            if missing:
                msg += "\n\nMissing/unreadable files:\n" + "\n".join(missing[:10])
            messagebox.showerror("Load session", msg)
            return

        self.structures = loaded
        ref_idx = int(session.get("reference_index", 0))
        self.reference_index = min(max(ref_idx, 0), len(self.structures) - 1)
        self._apply_display_options(session.get("display_options", {}))
        self._cfg["last_session_path"] = path

        self._refresh_structure_list()
        self.redraw()
        self._refresh_analysis()

        if missing:
            messagebox.showwarning(
                "Load session",
                "Session loaded, but some files were missing or unreadable:\n\n"
                + "\n".join(missing[:10])
                + ("\n..." if len(missing) > 10 else ""),
            )
        self._set_status(f"Loaded session: {os.path.basename(path)}  ({len(loaded)} structure(s))")

    def _collect_config(self) -> dict:
        try:
            sash = self._paned.sash_coord(0)[0]
        except Exception:
            sash = self._cfg.get("sash_pos", 820)
        cfg = {
            "metal_index": "1",
            "use_manual_metal": False,
            "mx": "0.0", "my": "0.0", "mz": "0.0",
            **self._current_display_options(),
            "last_session_path": self._cfg.get("last_session_path", ""),
            "window_geometry":  self.root.geometry(),
            "sash_pos":         sash,
        }
        return cfg

    def _on_close(self):
        save_config(self._collect_config())
        if self._standalone:
            self.root.destroy()
        else:
            self.root.destroy()


# ─────────────────────────────────────────────────────────────────────────────
# Plugin / public entry points
# ─────────────────────────────────────────────────────────────────────────────

def open_coordination_polar_viewer(parent: Optional[tk.Misc] = None) -> CoordinationPolarViewer:
    return CoordinationPolarViewer(master=parent)


def open_plugin(app) -> CoordinationPolarViewer:
    return open_coordination_polar_viewer(parent=app.root)


def register(app):
    app.add_menu_item(
        label="Coordination Polar Viewer…",
        command=lambda: open_plugin(app),
    )


def main():
    viewer = CoordinationPolarViewer()
    viewer.root.mainloop()


if __name__ == "__main__":
    main()
