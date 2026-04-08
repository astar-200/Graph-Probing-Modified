from absl import app, flags
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

flags.DEFINE_string("input_file", None, "Path to the correlation matrix (.npy file)")
flags.DEFINE_string("output_file", None, "Output path for the heatmap image. If None, auto-generate based on input file.")
flags.DEFINE_string("title", None, "Title for the heatmap. If None, use filename.")
flags.DEFINE_string("cmap", "RdBu_r", "Colormap for the heatmap (default: RdBu_r for correlation matrices)")
flags.DEFINE_float("vmin", None, "Minimum value for colorbar. If None, use data min.")
flags.DEFINE_float("vmax", None, "Maximum value for colorbar. If None, use data max.")
flags.DEFINE_boolean("center_zero", True, "Whether to center the colormap at zero (useful for correlation matrices)")
flags.DEFINE_integer("figsize", 12, "Figure size (width and height in inches)")
flags.DEFINE_integer("dpi", 300, "DPI for saved image")
flags.DEFINE_boolean("show_colorbar", True, "Whether to show colorbar")
flags.DEFINE_boolean("show_ticks", False, "Whether to show axis ticks (not recommended for large matrices)")
flags.DEFINE_string("format", "png", "Output format (png, pdf, svg, jpg)")
flags.DEFINE_float("threshold", None, "Threshold for filtering: keep top threshold proportion of elements by absolute value (e.g., 0.1 means keep top 10%%). If None, no filtering applied.")
FLAGS = flags.FLAGS


def load_correlation_matrix(filepath):
    """Load correlation matrix from .npy file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    corr_matrix = np.load(filepath)
    print(f"Loaded correlation matrix with shape: {corr_matrix.shape}")
    print(f"Value range: [{np.min(corr_matrix):.4f}, {np.max(corr_matrix):.4f}]")
    print(f"Mean: {np.mean(corr_matrix):.4f}, Std: {np.std(corr_matrix):.4f}")
    
    return corr_matrix


def plot_heatmap(corr_matrix, title, cmap, vmin, vmax, center_zero, figsize, 
                 show_colorbar, show_ticks, output_file, dpi, fmt, threshold=None):
    """Generate and save heatmap."""
    
    # Create a copy for visualization
    corr_matrix_vis = corr_matrix.copy()
    
    # Apply threshold filtering if specified
    if threshold is not None and 0 < threshold <= 1.0:
        # Calculate percentile threshold: keep top threshold proportion
        percentile_threshold = (1 - threshold) * 100
        threshold_value = np.percentile(np.abs(corr_matrix_vis), percentile_threshold)
        
        # Set elements with absolute value below threshold to 0
        mask = np.abs(corr_matrix_vis) < threshold_value
        corr_matrix_vis[mask] = 0
        
        num_kept = np.sum(np.abs(corr_matrix_vis) > 0)
        total = corr_matrix_vis.size
        print(f"Applied threshold filtering (threshold={threshold:.2%}): kept {num_kept}/{total} elements ({num_kept/total:.2%})")
        print(f"Threshold value: {threshold_value:.6f}")
    
    # Set diagonal to 0 for visualization
    np.fill_diagonal(corr_matrix_vis, 0)
    print(f"Diagonal elements set to 0 for visualization")
    print(f"Off-diagonal value range: [{np.min(corr_matrix_vis):.4f}, {np.max(corr_matrix_vis):.4f}]")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    
    # Determine vmin and vmax from off-diagonal elements
    if vmin is None:
        vmin = np.min(corr_matrix_vis)
    if vmax is None:
        vmax = np.max(corr_matrix_vis)
    
    # Create normalization
    if center_zero:
        # Center colormap at zero
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    else:
        norm = None
    
    # Plot heatmap using the visualization matrix (with diagonal set to 0)
    im = ax.imshow(corr_matrix_vis, cmap=cmap, aspect='auto', norm=norm, 
                   interpolation='nearest')
    
    # Add colorbar
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Correlation', rotation=270, labelpad=20, fontsize=12)
    
    # Set title
    if title:
        ax.set_title(title, fontsize=14, pad=20)
    
    # Configure ticks
    if show_ticks:
        ax.set_xlabel('Neuron Index', fontsize=12)
        ax.set_ylabel('Neuron Index', fontsize=12)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        # Still show labels
        ax.set_xlabel('Neuron Index', fontsize=12)
        ax.set_ylabel('Neuron Index', fontsize=12)
    
    # Add grid lines for better visualization (optional, only for smaller matrices)
    if corr_matrix_vis.shape[0] <= 50:
        ax.set_xticks(np.arange(corr_matrix_vis.shape[1])-.5, minor=True)
        ax.set_yticks(np.arange(corr_matrix_vis.shape[0])-.5, minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_file, dpi=dpi, format=fmt, bbox_inches='tight')
    print(f"Heatmap saved to: {output_file}")
    
    plt.close()


def main(_):
    if FLAGS.input_file is None:
        print("Error: --input_file is required")
        print("\nExample usage:")
        print("  python -m graph_probing.visualize_correlation_matrix \\")
        print("    --input_file data/graph_probing/sum/Qwen2.5-0.5B-dense/average_graphs/layer_12_avg_corr.npy")
        return
    
    # Load correlation matrix
    corr_matrix = load_correlation_matrix(FLAGS.input_file)
    
    # Determine output file
    if FLAGS.output_file:
        output_file = FLAGS.output_file
    else:
        # Auto-generate output filename
        input_dir = os.path.dirname(FLAGS.input_file)
        input_basename = os.path.basename(FLAGS.input_file)
        output_basename = input_basename.replace('.npy', f'_heatmap.{FLAGS.format}')
        output_file = os.path.join(input_dir, output_basename)
    
    # Determine title
    if FLAGS.title:
        title = FLAGS.title
    else:
        # Extract meaningful title from filename
        basename = os.path.basename(FLAGS.input_file)
        title = basename.replace('.npy', '').replace('_', ' ').title()
    
    # Generate heatmap
    print(f"\nGenerating heatmap...")
    print(f"  Colormap: {FLAGS.cmap}")
    print(f"  Figure size: {FLAGS.figsize}x{FLAGS.figsize} inches")
    print(f"  DPI: {FLAGS.dpi}")
    print(f"  Center at zero: {FLAGS.center_zero}")
    if FLAGS.threshold is not None:
        print(f"  Threshold: {FLAGS.threshold:.2%} (keep top {FLAGS.threshold:.2%} by absolute value)")
    
    plot_heatmap(
        corr_matrix=corr_matrix,
        title=title,
        cmap=FLAGS.cmap,
        vmin=FLAGS.vmin,
        vmax=FLAGS.vmax,
        center_zero=FLAGS.center_zero,
        figsize=FLAGS.figsize,
        show_colorbar=FLAGS.show_colorbar,
        show_ticks=FLAGS.show_ticks,
        output_file=output_file,
        dpi=FLAGS.dpi,
        fmt=FLAGS.format,
        threshold=FLAGS.threshold
    )
    
    print("\nVisualization completed!")


if __name__ == "__main__":
    app.run(main)

