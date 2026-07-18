"""
Thesis Figure: Manipulator Keyframing — Safe vs Unsafe Trajectory.

Generates 6 keyframe images (3 safe + 3 unsafe) showing the Panda arm
navigating around a cylindrical obstacle.

Uses the FK chain to compute all joint positions, then renders the arm
as thick 3D links with an obstacle cylinder.

Output: thesis_figures/keyframe_safe_1.png ... keyframe_unsafe_3.png

Usage:
    python thesis_keyframes.py
"""

import numpy as np
import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from sim.panda import Panda
from sim.transform_matrix import z_rotation_matrix

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'thesis_figures')


# =============================================================================
# Forward Kinematics — compute ALL joint positions (not just end-effector)
# =============================================================================
def compute_joint_positions(q_deg):
    """
    Compute 3D positions of all 8 joint frames (base + 7 joints + EE)
    using the Panda DH parameters.

    Args:
        q_deg: list/array of 7 joint angles in DEGREES

    Returns:
        positions: (9, 3) array — base, 7 joints, end-effector
    """
    # Panda DH parameters: (a, d, alpha, theta_offset)
    # From franka documentation
    dh_params = [
        (0,       0.333,  0,    0),     # Joint 1
        (0,       0,     -90,   0),     # Joint 2
        (0,       0.316,  90,   0),     # Joint 3
        (0.0825,  0,      90,   0),     # Joint 4
        (-0.0825, 0.384, -90,   0),     # Joint 5
        (0,       0,      90,   0),     # Joint 6
        (0.088,   0,      90,   0),     # Joint 7
    ]
    ee_offset = np.array([0, 0, 0.107 + 0.0584])

    positions = [np.array([0.0, 0.0, 0.0])]  # base at origin

    T = np.eye(4)
    for i, (a, d, alpha, theta_off) in enumerate(dh_params):
        theta = q_deg[i] + theta_off
        theta_rad = np.radians(theta)
        alpha_rad = np.radians(alpha)

        # Standard DH transform: Rz(theta) * Tz(d) * Tx(a) * Rx(alpha)
        ct, st = np.cos(theta_rad), np.sin(theta_rad)
        ca, sa = np.cos(alpha_rad), np.sin(alpha_rad)

        T_i = np.array([
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [0,   sa,       ca,      d     ],
            [0,   0,        0,       1     ],
        ])
        T = T @ T_i
        positions.append(T[:3, 3].copy())

    # End-effector
    ee_pos = T @ np.append(ee_offset, 1)
    positions.append(ee_pos[:3])

    return np.array(positions)


# =============================================================================
# Drawing helpers
# =============================================================================
def draw_cylinder(ax, x, y, h, r, color='red', alpha=0.3):
    """Draw a cylinder obstacle."""
    theta = np.linspace(0, 2 * np.pi, 40)
    z_cyl = np.linspace(0, h, 20)
    theta_grid, z_grid = np.meshgrid(theta, z_cyl)
    x_cyl = x + r * np.cos(theta_grid)
    y_cyl = y + r * np.sin(theta_grid)
    ax.plot_surface(x_cyl, y_cyl, z_grid, alpha=alpha, color=color)

    # Top and bottom caps
    r_cap = np.linspace(0, r, 5)
    for z_val in [0, h]:
        for ri in r_cap:
            ax.plot(x + ri * np.cos(theta), y + ri * np.sin(theta),
                    z_val * np.ones_like(theta), color=color, alpha=alpha * 0.5,
                    linewidth=0.3)


def draw_robot(ax, joint_positions, color='#2196F3', linewidth=6, alpha=1.0,
               label=None):
    """Draw the robot arm as thick connected links."""
    pos = joint_positions

    # Draw links
    for i in range(len(pos) - 1):
        ax.plot([pos[i, 0], pos[i+1, 0]],
                [pos[i, 1], pos[i+1, 1]],
                [pos[i, 2], pos[i+1, 2]],
                '-', color=color, linewidth=linewidth, alpha=alpha,
                solid_capstyle='round', label=label if i == 0 else None)

    # Draw joints as spheres
    for i in range(len(pos)):
        size = 80 if i == 0 else (60 if i < len(pos) - 1 else 40)
        jcolor = '#333333' if i < len(pos) - 1 else '#FF9800'
        ax.scatter(*pos[i], s=size, color=jcolor, zorder=5,
                   edgecolors='black', linewidth=0.5)

    # End-effector marker
    ax.scatter(*pos[-1], s=100, color='#FF9800', marker='D', zorder=10,
               edgecolors='black', linewidth=0.8)


def draw_base(ax):
    """Draw a base plate."""
    plate_size = 0.15
    x = [-plate_size, plate_size, plate_size, -plate_size]
    y = [-plate_size, -plate_size, plate_size, plate_size]
    z = [0, 0, 0, 0]
    verts = [list(zip(x, y, z))]
    poly = Poly3DCollection(verts, alpha=0.4, facecolor='#90A4AE',
                            edgecolor='#546E7A', linewidth=1.5)
    ax.add_collection3d(poly)


def setup_ax(ax, title, elev=25, azim=-60):
    """Configure 3D axes."""
    ax.set_xlim(-0.4, 0.9)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.05, 1.1)
    ax.set_xlabel('X (m)', fontsize=10, labelpad=5)
    ax.set_ylabel('Y (m)', fontsize=10, labelpad=5)
    ax.set_zlabel('Z (m)', fontsize=10, labelpad=5)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.view_init(elev=elev, azim=azim)
    ax.tick_params(labelsize=8)
    # Light background
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')


# =============================================================================
# Generate keyframe configurations
# =============================================================================
def get_obstacle():
    """Fixed obstacle: cylinder at (0.5, 0.0) with h=0.75, r=0.1"""
    return [0.5, 0.0, 0.75, 0.1]


def get_safe_trajectory():
    """
    3 keyframe configs where the arm goes AROUND the obstacle.
    Hand-picked joint angles (degrees) for a visually clear safe path.
    """
    configs = [
        # Frame 1: Start — arm extended, away from obstacle
        [-30, -40, 30, -120, 0, 80, 0],
        # Frame 2: Mid — arm curves around the obstacle (to the side)
        [30, -30, 10, -90, -40, 100, 30],
        # Frame 3: Goal — arm reaches target on the other side
        [60, -20, -10, -80, 10, 90, 45],
    ]
    return configs


def get_unsafe_trajectory():
    """
    3 keyframe configs where the arm passes THROUGH the obstacle.
    The middle frame should clearly intersect the cylinder.
    """
    configs = [
        # Frame 1: Start — same as safe start
        [-30, -40, 30, -120, 0, 80, 0],
        # Frame 2: Mid — arm goes straight through obstacle!
        [20, -50, 0, -100, 0, 70, 0],
        # Frame 3: Goal — same as safe goal
        [60, -20, -10, -80, 10, 90, 45],
    ]
    return configs


def refine_configs_with_collision_check():
    """
    Use Robo3D to verify which configs actually collide.
    Adjust the unsafe mid config until it genuinely collides.
    """
    from sim.robot3d import Robo3D
    robo3d = Robo3D(Panda())
    obstacle = get_obstacle()

    # Test safe configs
    safe_configs = get_safe_trajectory()
    for i, q in enumerate(safe_configs):
        collides = robo3d.check_for_collision(q, [obstacle])
        dist = robo3d.dist_jpos_to_obstacles(q, [obstacle])
        print(f"Safe frame {i+1}: collides={collides}, dist={dist:.4f}")
        if collides:
            print(f"  WARNING: Safe frame {i+1} collides! Needs adjustment.")

    # Test unsafe configs
    unsafe_configs = get_unsafe_trajectory()
    for i, q in enumerate(unsafe_configs):
        collides = robo3d.check_for_collision(q, [obstacle])
        dist = robo3d.dist_jpos_to_obstacles(q, [obstacle])
        print(f"Unsafe frame {i+1}: collides={collides}, dist={dist:.4f}")

    return safe_configs, unsafe_configs


def find_colliding_config(robo3d, obstacle, base_config, n_search=500):
    """Search for a config near base_config that actually collides."""
    base = np.array(base_config, dtype=float)
    for _ in range(n_search):
        noise = np.random.randn(7) * 10  # ±10 degree perturbation
        test = (base + noise).tolist()
        if robo3d.check_for_collision(test, [obstacle]):
            return test
    return None


def find_safe_configs(robo3d, obstacle, n_search=1000):
    """
    Search for 3 safe configs that form a visually clear trajectory
    around the obstacle.
    """
    robot = Panda()
    robot.to('cpu')
    q_min = robot.joint_min_limits_tensor.numpy()
    q_max = robot.joint_max_limits_tensor.numpy()

    safe_configs = []
    for _ in range(n_search):
        q = np.random.uniform(q_min, q_max)
        if not robo3d.check_for_collision(q.tolist(), [obstacle]):
            pos = compute_joint_positions(q.tolist())
            ee = pos[-1]
            # Want EE roughly near the obstacle height but to the side
            if 0.3 < ee[2] < 0.9 and abs(ee[0] - 0.5) < 0.4:
                safe_configs.append(q.tolist())
                if len(safe_configs) >= 10:
                    break
    return safe_configs


# =============================================================================
# Main: Generate keyframe images
# =============================================================================
def main():
    from sim.robot3d import Robo3D

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    obstacle = get_obstacle()
    ox, oy, oh, or_ = obstacle

    robo3d = Robo3D(Panda())

    # Get and verify configs
    print("Verifying configurations...")
    safe_configs = get_safe_trajectory()
    unsafe_configs = get_unsafe_trajectory()

    # Check and fix
    for i, q in enumerate(safe_configs):
        if robo3d.check_for_collision(q, [obstacle]):
            print(f"Safe frame {i+1} collides — searching for alternative...")
            # Search nearby
            for attempt in range(1000):
                q_test = (np.array(q) + np.random.randn(7) * 5).tolist()
                if not robo3d.check_for_collision(q_test, [obstacle]):
                    safe_configs[i] = q_test
                    print(f"  Found safe alternative at attempt {attempt}")
                    break

    # Ensure unsafe mid-frame collides
    if not robo3d.check_for_collision(unsafe_configs[1], [obstacle]):
        print("Unsafe mid-frame doesn't collide — searching...")
        result = find_colliding_config(robo3d, obstacle, unsafe_configs[1], n_search=2000)
        if result:
            unsafe_configs[1] = result
            print("  Found colliding config!")
        else:
            # Try more aggressive search
            print("  Broadening search...")
            for attempt in range(5000):
                q_test = np.random.uniform(-100, 100, 7).tolist()
                if robo3d.check_for_collision(q_test, [obstacle]):
                    # Verify it looks reasonable
                    pos = compute_joint_positions(q_test)
                    if pos[-1][2] > 0.2:  # EE above ground
                        unsafe_configs[1] = q_test
                        print(f"  Found at attempt {attempt}")
                        break

    # Print verification
    print("\nFinal verification:")
    for i, q in enumerate(safe_configs):
        col = robo3d.check_for_collision(q, [obstacle])
        dist = robo3d.dist_jpos_to_obstacles(q, [obstacle])
        print(f"  Safe {i+1}: collides={col}, dist={dist:.4f}")
    for i, q in enumerate(unsafe_configs):
        col = robo3d.check_for_collision(q, [obstacle])
        dist = robo3d.dist_jpos_to_obstacles(q, [obstacle])
        print(f"  Unsafe {i+1}: collides={col}, dist={dist:.4f}")

    # =========================================================================
    # Render keyframes
    # =========================================================================
    frame_labels = ['Start', 'Mid-trajectory', 'Goal reached']
    elev, azim = 28, -55

    # --- Safe trajectory keyframes ---
    for i, q_deg in enumerate(safe_configs):
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection='3d')

        joint_pos = compute_joint_positions(q_deg)
        draw_base(ax)
        draw_cylinder(ax, ox, oy, oh, or_, color='#F44336', alpha=0.25)
        draw_robot(ax, joint_pos, color='#2196F3', linewidth=5)

        # Draw ghost of EE path
        if i > 0:
            for j in range(i):
                prev_pos = compute_joint_positions(safe_configs[j])
                ax.scatter(*prev_pos[-1], s=40, color='#90CAF9', alpha=0.5,
                           marker='o', zorder=3)
            # Connect EE positions with dashed line
            ee_positions = [compute_joint_positions(safe_configs[j])[-1]
                           for j in range(i + 1)]
            ee_arr = np.array(ee_positions)
            ax.plot(ee_arr[:, 0], ee_arr[:, 1], ee_arr[:, 2],
                    '--', color='#1565C0', linewidth=2, alpha=0.6)

        # Green checkmark annotation
        title = f'Safe Trajectory — {frame_labels[i]}'
        setup_ax(ax, title, elev=elev, azim=azim)

        # Green border
        for spine in ax.spines.values():
            spine.set_edgecolor('#4CAF50')
            spine.set_linewidth(3)

        plt.tight_layout()
        fname = os.path.join(OUTPUT_DIR, f'keyframe_safe_{i+1}.png')
        plt.savefig(fname, dpi=200, bbox_inches='tight',
                    facecolor='white', edgecolor='#4CAF50')
        plt.close()
        print(f"Saved: {fname}")

    # --- Unsafe trajectory keyframes ---
    for i, q_deg in enumerate(unsafe_configs):
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection='3d')

        joint_pos = compute_joint_positions(q_deg)
        draw_base(ax)
        draw_cylinder(ax, ox, oy, oh, or_, color='#F44336', alpha=0.25)

        # For the collision frame, highlight in red
        arm_color = '#F44336' if i == 1 else '#2196F3'
        draw_robot(ax, joint_pos, color=arm_color, linewidth=5)

        # Ghost trail
        if i > 0:
            for j in range(i):
                prev_pos = compute_joint_positions(unsafe_configs[j])
                ax.scatter(*prev_pos[-1], s=40, color='#EF9A9A', alpha=0.5,
                           marker='o', zorder=3)
            ee_positions = [compute_joint_positions(unsafe_configs[j])[-1]
                           for j in range(i + 1)]
            ee_arr = np.array(ee_positions)
            ax.plot(ee_arr[:, 0], ee_arr[:, 1], ee_arr[:, 2],
                    '--', color='#C62828', linewidth=2, alpha=0.6)

        # Title with collision indicator
        if i == 1:
            title = f'Unsafe Trajectory — {frame_labels[i]} ⚠ COLLISION'
        else:
            title = f'Unsafe Trajectory — {frame_labels[i]}'
        setup_ax(ax, title, elev=elev, azim=azim)

        # Red border
        for spine in ax.spines.values():
            spine.set_edgecolor('#F44336')
            spine.set_linewidth(3)

        plt.tight_layout()
        fname = os.path.join(OUTPUT_DIR, f'keyframe_unsafe_{i+1}.png')
        plt.savefig(fname, dpi=200, bbox_inches='tight',
                    facecolor='white', edgecolor='#F44336')
        plt.close()
        print(f"Saved: {fname}")

    # =========================================================================
    # Also generate combined 2x3 grid
    # =========================================================================
    fig, axes_grid = plt.subplots(2, 3, figsize=(22, 14),
                                   subplot_kw={'projection': '3d'})

    for col in range(3):
        # Row 0: Safe
        ax = axes_grid[0, col]
        q_deg = safe_configs[col]
        joint_pos = compute_joint_positions(q_deg)
        draw_base(ax)
        draw_cylinder(ax, ox, oy, oh, or_, color='#F44336', alpha=0.2)
        draw_robot(ax, joint_pos, color='#2196F3', linewidth=4)
        if col > 0:
            ee_trail = [compute_joint_positions(safe_configs[j])[-1]
                       for j in range(col + 1)]
            ee_arr = np.array(ee_trail)
            ax.plot(ee_arr[:, 0], ee_arr[:, 1], ee_arr[:, 2],
                    '--', color='#1565C0', linewidth=1.5, alpha=0.5)
        setup_ax(ax, f'✅ {frame_labels[col]}', elev=elev, azim=azim)

        # Row 1: Unsafe
        ax = axes_grid[1, col]
        q_deg = unsafe_configs[col]
        joint_pos = compute_joint_positions(q_deg)
        draw_base(ax)
        draw_cylinder(ax, ox, oy, oh, or_, color='#F44336', alpha=0.2)
        arm_color = '#F44336' if col == 1 else '#2196F3'
        draw_robot(ax, joint_pos, color=arm_color, linewidth=4)
        if col > 0:
            ee_trail = [compute_joint_positions(unsafe_configs[j])[-1]
                       for j in range(col + 1)]
            ee_arr = np.array(ee_trail)
            ax.plot(ee_arr[:, 0], ee_arr[:, 1], ee_arr[:, 2],
                    '--', color='#C62828', linewidth=1.5, alpha=0.5)
        label = f'❌ {frame_labels[col]}'
        if col == 1:
            label += ' ⚠'
        setup_ax(ax, label, elev=elev, azim=azim)

    # Row labels
    fig.text(0.02, 0.72, 'SAFE\nTrajectory', fontsize=16, fontweight='bold',
             color='#4CAF50', va='center', ha='center', rotation=90)
    fig.text(0.02, 0.28, 'UNSAFE\nTrajectory', fontsize=16, fontweight='bold',
             color='#F44336', va='center', ha='center', rotation=90)

    plt.suptitle('Manipulator Trajectory: Safe vs Unsafe Path',
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0.04, 0, 1, 0.96])
    fname = os.path.join(OUTPUT_DIR, 'keyframes_combined.png')
    plt.savefig(fname, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved combined grid: {fname}")

    print(f"\nAll keyframes saved to: {OUTPUT_DIR}")
    print("Individual files: keyframe_safe_1..3.png, keyframe_unsafe_1..3.png")
    print("Combined grid: keyframes_combined.png")


if __name__ == '__main__':
    main()
