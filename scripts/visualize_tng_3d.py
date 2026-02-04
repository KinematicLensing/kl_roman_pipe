#!/usr/bin/env python3
"""
Interactive 3D visualization of TNG50 galaxy particle distributions.

This script creates linked 3D scatter plots showing stellar particles, gas particles,
and SFR-weighted gas. All three panels rotate together to help understand the
gas-stellar angular momentum offset.

Usage:
    # Interactive mode
    conda run -n klpipe python scripts/visualize_tng_3d.py --subhalo-id 20
    conda run -n klpipe python scripts/visualize_tng_3d.py --index 3

    # Animation mode (360° rotation around edge-on stellar view)
    conda run -n klpipe python scripts/visualize_tng_3d.py --subhalo-id 20 --animate --output movie.gif
    conda run -n klpipe python scripts/visualize_tng_3d.py --subhalo-id 20 --animate --output movie.mp4 --fps 30
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from kl_pipe.tng import TNG50MockData


class Interactive3DGalaxyViewer:
    """Interactive 3D viewer for TNG50 galaxies with linked panels."""

    def __init__(self, galaxy_data, subhalo_id=None):
        """
        Initialize viewer with galaxy data.

        Parameters
        ----------
        galaxy_data : dict
            Dictionary with 'gas', 'stellar', 'subhalo' keys
        subhalo_id : int, optional
            SubhaloID for title
        """
        self.galaxy_data = galaxy_data
        self.stellar = galaxy_data['stellar']
        self.gas = galaxy_data['gas']
        self.subhalo = galaxy_data['subhalo']
        self.subhalo_id = subhalo_id or self.subhalo['SubhaloID']

        # Compute angular momentum vectors for reference
        self._compute_angular_momenta()

        # Create figure with 3 subplots
        self.fig = plt.figure(figsize=(18, 6))
        self.ax_stellar = self.fig.add_subplot(131, projection='3d')
        self.ax_gas = self.fig.add_subplot(132, projection='3d')
        self.ax_sfr = self.fig.add_subplot(133, projection='3d')

        # Store axes for syncing
        self.axes = [self.ax_stellar, self.ax_gas, self.ax_sfr]

        # Initialize plots
        self._setup_stellar_plot()
        self._setup_gas_plot()
        self._setup_sfr_plot()

        # Add angular momentum vector arrows
        self._add_L_vectors()

        # Set up synchronized view control
        self._setup_view_sync()

        # Add controls
        self._setup_controls()

        plt.tight_layout()

    def _compute_angular_momenta(self):
        """Compute angular momentum vectors for stellar and gas."""
        # Stellar: luminosity-weighted L
        coords_s = self.stellar['Coordinates']
        vel_s = self.stellar['Velocities']

        if 'Dusted_Luminosity_r' in self.stellar:
            weights_s = self.stellar['Dusted_Luminosity_r']
        elif 'Masses' in self.stellar:
            weights_s = self.stellar['Masses']
        else:
            weights_s = np.ones(len(coords_s))

        center_s = np.average(coords_s, axis=0, weights=weights_s)
        coords_s_cen = coords_s - center_s
        vel_s_mean = np.average(vel_s, axis=0, weights=weights_s)
        vel_s_cen = vel_s - vel_s_mean

        L_s_vec = np.sum(weights_s[:, None] * np.cross(coords_s_cen, vel_s_cen), axis=0)
        L_s_unit = L_s_vec / np.linalg.norm(L_s_vec)

        # Flip L if inclination >90° (catalog convention: inc>90° means L points downward)
        # We want L to point upward (+Z-ish) for intuitive edge-on views
        if self.subhalo['Inclination_star'] > 90:
            L_s_unit = -L_s_unit

        self.L_stellar = L_s_unit
        self.center_stellar = center_s

        # Gas: mass-weighted L
        coords_g = self.gas['Coordinates']
        vel_g = self.gas['Velocities']
        masses_g = self.gas['Masses']

        center_g = np.average(coords_g, axis=0, weights=masses_g)
        coords_g_cen = coords_g - center_g
        vel_g_mean = np.average(vel_g, axis=0, weights=masses_g)
        vel_g_cen = vel_g - vel_g_mean

        L_g_vec = np.sum(masses_g[:, None] * np.cross(coords_g_cen, vel_g_cen), axis=0)
        L_g_unit = L_g_vec / np.linalg.norm(L_g_vec)

        # Flip gas L if inclination >90°
        if self.subhalo['Inclination_gas'] > 90:
            L_g_unit = -L_g_unit

        self.L_gas = L_g_unit
        self.center_gas = center_g

        # Compute offset angle (preserves physical offset after flipping)
        self.L_offset_deg = np.rad2deg(
            np.arccos(np.clip(np.dot(self.L_stellar, self.L_gas), -1, 1))
        )

        # Compute single rotation matrix to align stellar L with +Z axis (Rodrigues formula)
        # Use same rotation for all panels so we can see the gas-stellar offset
        self.R = self._compute_rotation_to_z(self.L_stellar)

    def _compute_rotation_to_z(self, L_vec):
        """Compute rotation matrix that aligns L_vec with +Z axis using Rodrigues formula."""
        target = np.array([0, 0, 1])

        # If already aligned, return identity
        if np.allclose(L_vec, target):
            return np.eye(3)

        # If anti-aligned, rotate 180° around X
        if np.allclose(L_vec, -target):
            return np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

        # Rodrigues formula: R = I + sin(θ)K + (1-cos(θ))K²
        # where K is the cross-product matrix of the rotation axis
        axis = np.cross(L_vec, target)
        axis = axis / np.linalg.norm(axis)  # Normalize

        cos_angle = np.dot(L_vec, target)
        sin_angle = np.sqrt(1 - cos_angle**2)

        # Cross-product matrix
        K = np.array(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
        )

        R = np.eye(3) + sin_angle * K + (1 - cos_angle) * (K @ K)
        return R

    def _setup_stellar_plot(self):
        """Set up stellar particle 3D scatter plot."""
        coords = self.stellar['Coordinates']

        # Center on luminosity-weighted centroid
        coords_cen = coords - self.center_stellar

        # Rotate using stellar-based rotation (aligns stellar L with +Z)
        coords_cen = (self.R @ coords_cen.T).T

        # Color by r-band luminosity
        if 'Dusted_Luminosity_r' in self.stellar:
            colors = np.log10(self.stellar['Dusted_Luminosity_r'] + 1)
        else:
            colors = 'blue'

        # Downsample more aggressively for performance
        n_particles = len(coords_cen)
        if n_particles > 5000:
            indices = np.random.choice(n_particles, 5000, replace=False)
            coords_plot = coords_cen[indices]
            colors_plot = colors[indices] if isinstance(colors, np.ndarray) else colors
        else:
            coords_plot = coords_cen
            colors_plot = colors

        self.ax_stellar.scatter(
            coords_plot[:, 0],
            coords_plot[:, 1],
            coords_plot[:, 2],
            c=colors_plot,
            cmap='viridis',
            alpha=0.3,
            s=1,
            label='Stellar',
            rasterized=True,  # Faster rendering
        )

        self.ax_stellar.set_xlabel('X (ckpc/h)')
        self.ax_stellar.set_ylabel('Y (ckpc/h)')
        self.ax_stellar.set_zlabel('Z (ckpc/h)')
        self.ax_stellar.set_title(
            f'Stellar Particles\nSubhaloID {self.subhalo_id}', y=0.85
        )

        # Set equal aspect ratio
        self._set_equal_aspect(self.ax_stellar, coords_cen)

    def _setup_gas_plot(self):
        """Set up gas particle 3D scatter plot."""
        coords = self.gas['Coordinates']
        masses = self.gas['Masses']

        # Center on mass-weighted centroid
        coords_cen = coords - self.center_gas

        # Rotate using same stellar-based rotation (gas L will be offset from +Z)
        coords_cen = (self.R @ coords_cen.T).T

        # Color by log mass
        colors = np.log10(masses)

        # Downsample more aggressively for performance
        n_particles = len(coords_cen)
        if n_particles > 5000:
            indices = np.random.choice(n_particles, 5000, replace=False)
            coords_plot = coords_cen[indices]
            colors_plot = colors[indices]
        else:
            coords_plot = coords_cen
            colors_plot = colors

        self.ax_gas.scatter(
            coords_plot[:, 0],
            coords_plot[:, 1],
            coords_plot[:, 2],
            c=colors_plot,
            cmap='plasma',
            alpha=0.3,
            s=1,
            label='Gas',
            rasterized=True,  # Faster rendering
        )

        self.ax_gas.set_xlabel('X (ckpc/h)')
        self.ax_gas.set_ylabel('Y (ckpc/h)')
        self.ax_gas.set_zlabel('Z (ckpc/h)')
        self.ax_gas.set_title(
            f'Gas Particles\nL offset: {self.L_offset_deg:.1f}°', y=0.85
        )

        self._set_equal_aspect(self.ax_gas, coords_cen)

    def _setup_sfr_plot(self):
        """Set up SFR-weighted gas 3D scatter plot."""
        coords = self.gas['Coordinates']
        sfr = self.gas['StarFormationRate']

        # Only plot star-forming gas
        sf_mask = sfr > 0
        coords_sf = coords[sf_mask]
        sfr_sf = sfr[sf_mask]

        if len(coords_sf) == 0:
            self.ax_sfr.text(
                0.5,
                0.5,
                0.5,
                'No star formation',
                ha='center',
                va='center',
                transform=self.ax_sfr.transAxes,
            )
            return

        # Center on gas centroid
        coords_cen = coords_sf - self.center_gas

        # Rotate using same stellar-based rotation (gas L will be offset from +Z)
        coords_cen = (self.R @ coords_cen.T).T

        # Color by log SFR
        colors = np.log10(sfr_sf)
        # Downsample more aggressively for performance
        n_particles = len(coords_cen)
        if n_particles > 5000:
            indices = np.random.choice(n_particles, 5000, replace=False)
            coords_plot = coords_cen[indices]
            colors_plot = colors[indices]
        else:
            coords_plot = coords_cen
            colors_plot = colors

        self.ax_sfr.scatter(
            coords_plot[:, 0],
            coords_plot[:, 1],
            coords_plot[:, 2],
            c=colors_plot,
            cmap='hot',
            alpha=0.4,
            s=2,
            label='SFR',
            rasterized=True,  # Faster rendering
        )

        self.ax_sfr.set_xlabel('X (ckpc/h)')
        self.ax_sfr.set_ylabel('Y (ckpc/h)')
        self.ax_sfr.set_zlabel('Z (ckpc/h)')
        self.ax_sfr.set_title(f'Star-Forming Gas\n{len(coords_sf)} particles', y=0.85)

        self._set_equal_aspect(self.ax_sfr, coords_cen)

    def _add_L_vectors(self):
        """Add angular momentum vector arrows to plots (in rotated frame)."""
        # Scale arrows to ~30% of plot size
        arrow_scale = 50  # ckpc/h

        # Both L vectors rotated by same matrix (stellar-based)
        L_stellar_rot = self.R @ self.L_stellar  # Should point to +Z
        L_gas_rot = self.R @ self.L_gas  # Offset from +Z by ~26.5°

        # Stellar L vector (cyan arrow, should point to +Z)
        self.ax_stellar.quiver(
            0,
            0,
            0,
            L_stellar_rot[0] * arrow_scale,
            L_stellar_rot[1] * arrow_scale,
            L_stellar_rot[2] * arrow_scale,
            color='cyan',
            arrow_length_ratio=0.2,
            linewidth=3,
            label='L_stellar',
        )

        # Gas L vector (red arrow, offset from +Z) on both gas panels
        for ax in [self.ax_gas, self.ax_sfr]:
            ax.quiver(
                0,
                0,
                0,
                L_gas_rot[0] * arrow_scale,
                L_gas_rot[1] * arrow_scale,
                L_gas_rot[2] * arrow_scale,
                color='red',
                arrow_length_ratio=0.2,
                linewidth=3,
                label='L_gas',
            )

        # Also add stellar L to gas plot for comparison
        self.ax_gas.quiver(
            0,
            0,
            0,
            self.L_stellar[0] * arrow_scale,
            self.L_stellar[1] * arrow_scale,
            self.L_stellar[2] * arrow_scale,
            color='cyan',
            arrow_length_ratio=0.2,
            linewidth=2,
            linestyle='--',
            alpha=0.5,
            label='L_stellar',
        )

    def _set_equal_aspect(self, ax, coords):
        """Set equal aspect ratio for 3D plot."""
        # Get coordinate ranges
        x_range = coords[:, 0].ptp()
        y_range = coords[:, 1].ptp()
        z_range = coords[:, 2].ptp()
        max_range = max(x_range, y_range, z_range)

        # Set limits centered on origin
        ax.set_xlim(-max_range / 2, max_range / 2)
        ax.set_ylim(-max_range / 2, max_range / 2)
        ax.set_zlim(-max_range / 2, max_range / 2)

    def _setup_view_sync(self):
        """Set up synchronized viewing angles across all three plots."""
        # Store initial view
        self.elev = 30
        self.azim = -60

        # Apply to all axes
        for ax in self.axes:
            ax.view_init(elev=self.elev, azim=self.azim)

        # Connect mouse events for manual rotation
        self.fig.canvas.mpl_connect('button_release_event', self._on_mouse_release)

    def _on_mouse_release(self, event):
        """Sync view angles when user rotates one plot."""
        # Find which axis was interacted with
        for ax in self.axes:
            if event.inaxes == ax:
                # Get new viewing angle
                self.elev = ax.elev
                self.azim = ax.azim

                # Apply to all other axes
                for other_ax in self.axes:
                    if other_ax != ax:
                        other_ax.view_init(elev=self.elev, azim=self.azim)

                # Update sliders if they exist
                if hasattr(self, 'slider_elev'):
                    self.slider_elev.set_val(self.elev)
                if hasattr(self, 'slider_azim'):
                    self.slider_azim.set_val(self.azim)

                self.fig.canvas.draw_idle()
                break

    def _setup_controls(self):
        """Add sliders and buttons for view control."""
        # Make room for controls (more space)
        self.fig.subplots_adjust(bottom=0.25, top=0.95)

        # Elevation slider
        ax_elev = self.fig.add_axes([0.15, 0.14, 0.3, 0.02])
        self.slider_elev = Slider(
            ax_elev, 'Elevation', -90, 90, valinit=self.elev, valstep=5
        )
        self.slider_elev.on_changed(self._update_view)

        # Azimuth slider
        ax_azim = self.fig.add_axes([0.15, 0.09, 0.3, 0.02])
        self.slider_azim = Slider(
            ax_azim, 'Azimuth', -180, 180, valinit=self.azim, valstep=5
        )
        self.slider_azim.on_changed(self._update_view)

        # Preset view buttons (more spaced out)
        ax_face = self.fig.add_axes([0.55, 0.13, 0.08, 0.04])
        self.btn_face = Button(ax_face, 'Face-on')
        self.btn_face.on_clicked(self._view_face_on)

        ax_edge = self.fig.add_axes([0.65, 0.13, 0.08, 0.04])
        self.btn_edge = Button(ax_edge, 'Edge-on')
        self.btn_edge.on_clicked(self._view_edge_on)

        ax_iso = self.fig.add_axes([0.75, 0.13, 0.08, 0.04])
        self.btn_iso = Button(ax_iso, 'Isometric')
        self.btn_iso.on_clicked(self._view_isometric)

        # Print info button
        ax_info = self.fig.add_axes([0.55, 0.07, 0.28, 0.04])
        self.btn_info = Button(ax_info, 'Print L vectors')
        self.btn_info.on_clicked(self._print_info)

    def _update_view(self, val=None):
        """Update view angles from sliders."""
        self.elev = self.slider_elev.val
        self.azim = self.slider_azim.val

        for ax in self.axes:
            ax.view_init(elev=self.elev, azim=self.azim)

        self.fig.canvas.draw_idle()

    def _view_face_on(self, event):
        """Set face-on view (looking down z-axis)."""
        self.slider_elev.set_val(90)
        self.slider_azim.set_val(0)

    def _view_edge_on(self, event):
        """Set edge-on view (looking along xy-plane)."""
        self.slider_elev.set_val(0)
        self.slider_azim.set_val(-90)

    def _view_isometric(self, event):
        """Set isometric view."""
        self.slider_elev.set_val(30)
        self.slider_azim.set_val(-60)

    def _print_info(self, event=None):
        """Print angular momentum information."""
        print("\n" + "=" * 60)
        print(f"SubhaloID {self.subhalo_id} Angular Momentum Information")
        print("=" * 60)
        print(
            f"\nStellar L vector: [{self.L_stellar[0]:.4f}, "
            f"{self.L_stellar[1]:.4f}, {self.L_stellar[2]:.4f}]"
        )
        print(
            f"Gas L vector:     [{self.L_gas[0]:.4f}, "
            f"{self.L_gas[1]:.4f}, {self.L_gas[2]:.4f}]"
        )
        print(f"\nL·L (dot product): {np.dot(self.L_stellar, self.L_gas):.4f}")
        print(f"Offset angle: {self.L_offset_deg:.2f}°")
        print(
            f"\nCatalog stellar inclination: "
            f"{self.subhalo['Inclination_star']:.2f}°"
        )
        print(f"Catalog gas inclination: " f"{self.subhalo['Inclination_gas']:.2f}°")
        print(
            f"Catalog PA difference: "
            f"{abs(self.subhalo['Position_Angle_star'] - self.subhalo['Position_Angle_gas']):.2f}°"
        )
        print("=" * 60 + "\n")

    def show(self):
        """Display the interactive viewer."""
        self._print_info()  # Print info on startup
        plt.show()

    def create_rotation_animation(
        self, output_path, n_frames=120, fps=30, start_elev=0, start_azim=-90
    ):
        """
        Create animation rotating around edge-on stellar view.

        Parameters
        ----------
        output_path : str or Path
            Output file path (.gif or .mp4)
        n_frames : int, default=120
            Number of frames (default = 120 frames at 30fps = 4 seconds)
        fps : int, default=30
            Frames per second
        start_elev : float, default=0
            Starting elevation (0 = edge-on)
        start_azim : float, default=-90
            Starting azimuth
        """
        output_path = Path(output_path)

        print(f"\nCreating {n_frames}-frame animation...")
        print(f"Starting view: elevation={start_elev}°, azimuth={start_azim}°")
        print(f"Rotating 360° around stellar disk...")

        # Remove interactive controls for animation
        for widget in [
            self.slider_elev,
            self.slider_azim,
            self.btn_face,
            self.btn_edge,
            self.btn_iso,
            self.btn_info,
        ]:
            widget.ax.set_visible(False)

        # Adjust layout - more top margin for title
        self.fig.subplots_adjust(bottom=0.05, top=0.88)

        # Add overall figure title with galaxy info
        # Position suptitle in the middle of the gap (0.88 to 1.0 = 12% space)
        self.fig.suptitle(
            f'TNG50 SubhaloID {self.subhalo_id} | '
            f'Gas-Stellar L Offset: {self.L_offset_deg:.1f}° | '
            f'Stellar Inc: {self.subhalo["Inclination_star"]:.1f}° | '
            f'Gas Inc: {self.subhalo["Inclination_gas"]:.1f}°',
            fontsize=14,
            fontweight='bold',
            y=0.94,
        )

        # Define animation function
        def update(frame):
            # Rotate azimuth 360° over n_frames
            azim = start_azim + (360 * frame / n_frames)

            # Update all axes
            for ax in self.axes:
                ax.view_init(elev=start_elev, azim=azim)

            # Update stellar panel title with current angle (keep y=0.85 to match other titles)
            self.ax_stellar.set_title(
                f'Stellar Particles\nAzimuth: {azim:.1f}°', y=0.85
            )

            return self.axes

        # Create animation
        anim = FuncAnimation(
            self.fig, update, frames=n_frames, interval=1000 / fps, blit=False
        )

        # Save animation
        if output_path.suffix == '.gif':
            print(f"Saving as GIF (this may take a while)...")
            writer = PillowWriter(fps=fps)
            anim.save(output_path, writer=writer, dpi=100)
        elif output_path.suffix == '.mp4':
            print(f"Saving as MP4...")
            # Higher bitrate for smoother playback, especially for face-on views
            writer = FFMpegWriter(
                fps=fps,
                bitrate=5000,
                codec='libx264',
                extra_args=['-pix_fmt', 'yuv420p'],
            )
            anim.save(output_path, writer=writer, dpi=100)
        else:
            raise ValueError(
                f"Unsupported format: {output_path.suffix}. Use .gif or .mp4"
            )

        print(f"✓ Animation saved to: {output_path}")
        print(f"  Duration: {n_frames/fps:.1f} seconds")
        print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Interactive 3D visualization of TNG50 galaxies'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--index', type=int, help='Galaxy array index (0-4)')
    group.add_argument(
        '--subhalo-id', type=int, help='SubhaloID (8, 17, 19, 20, or 29)'
    )

    # Animation options
    parser.add_argument(
        '--animate',
        action='store_true',
        help='Create animation instead of interactive viewer',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='tng_rotation.gif',
        help='Output file path for animation (.gif or .mp4)',
    )
    parser.add_argument(
        '--n-frames',
        type=int,
        default=120,
        help='Number of animation frames (default: 120)',
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Frames per second for animation (default: 30)',
    )
    parser.add_argument(
        '--start-elev',
        type=float,
        default=0,
        help='Starting elevation angle (default: 0 = edge-on)',
    )
    parser.add_argument(
        '--start-azim',
        type=float,
        default=-90,
        help='Starting azimuth angle (default: -90)',
    )

    args = parser.parse_args()

    # Load data
    print("Loading TNG50 data...")
    tng_data = TNG50MockData()

    if args.subhalo_id is not None:
        galaxy = tng_data.get_galaxy(subhalo_id=args.subhalo_id)
        subhalo_id = args.subhalo_id
    else:
        galaxy = tng_data.get_galaxy(index=args.index)
        subhalo_id = galaxy['subhalo']['SubhaloID']

    # Create viewer
    viewer = Interactive3DGalaxyViewer(galaxy, subhalo_id=subhalo_id)

    if args.animate:
        # Animation mode
        print(f"\nCreating animation for SubhaloID {subhalo_id}...")
        viewer.create_rotation_animation(
            output_path=args.output,
            n_frames=args.n_frames,
            fps=args.fps,
            start_elev=args.start_elev,
            start_azim=args.start_azim,
        )
        plt.close()
    else:
        # Interactive mode
        print(f"Visualizing SubhaloID {subhalo_id}...")
        print("\nInstructions:")
        print("- Click and drag any plot to rotate (all three sync)")
        print("- Use sliders to adjust elevation/azimuth")
        print("- Click preset buttons for standard views")
        print("- Cyan arrow = stellar L, Red arrow = gas L")
        print("- Close window to exit\n")
        viewer.show()


if __name__ == '__main__':
    main()
