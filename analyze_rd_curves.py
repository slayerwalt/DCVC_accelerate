#!/usr/bin/env python3
"""
Script to analyze video compression results and generate RD curves (PSNR-BPP).
Analyzes JSON result files and plots average PSNR vs BPP for different rate points.
"""

import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_json_results(json_path):
    """Load results from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def extract_rd_points(data, dataset_name):
    """
    Extract RD points (PSNR, BPP) and encoding/decoding times for each rate index from the dataset.
    
    Args:
        data: Dictionary containing the loaded JSON data
        dataset_name: Name of the dataset (e.g., "UVG")
    
    Returns:
        Tuple of (rate_data, timing_data) where:
        - rate_data: Dictionary mapping rate_idx to lists of PSNR and BPP values
        - timing_data: Dictionary with encoding and decoding time lists
    """
    if dataset_name not in data:
        print(f"Warning: Dataset '{dataset_name}' not found in results")
        return {}, {'encoding': [], 'decoding': []}
    
    dataset = data[dataset_name]
    
    # Group by rate index
    rate_data = defaultdict(lambda: {'psnr': [], 'bpp': []})
    timing_data = {'encoding': [], 'decoding': []}
    
    for video_name, video_data in dataset.items():
        for rate_idx, rate_info in video_data.items():
            psnr = rate_info.get('ave_all_frame_psnr', None)
            bpp = rate_info.get('ave_all_frame_bpp', None)
            
            if psnr is not None and bpp is not None:
                rate_data[rate_idx]['psnr'].append(psnr)
                rate_data[rate_idx]['bpp'].append(bpp)
            
            # Collect encoding and decoding times
            enc_time = rate_info.get('avg_frame_encoding_time', None)
            dec_time = rate_info.get('avg_frame_decoding_time', None)
            
            if enc_time is not None:
                timing_data['encoding'].append(enc_time)
            if dec_time is not None:
                timing_data['decoding'].append(dec_time)
    
    return rate_data, timing_data


def compute_average_rd_points(rate_data):
    """
    Compute average PSNR and BPP for each rate index.
    
    Args:
        rate_data: Dictionary mapping rate_idx to lists of PSNR and BPP values
    
    Returns:
        Two lists: average_bpp and average_psnr, sorted by BPP
    """
    avg_points = []
    
    for rate_idx in sorted(rate_data.keys()):
        psnr_values = rate_data[rate_idx]['psnr']
        bpp_values = rate_data[rate_idx]['bpp']
        
        if len(psnr_values) > 0 and len(bpp_values) > 0:
            avg_psnr = np.mean(psnr_values)
            avg_bpp = np.mean(bpp_values)
            avg_points.append((avg_bpp, avg_psnr))
    
    # Sort by BPP
    avg_points.sort(key=lambda x: x[0])
    
    avg_bpp = [p[0] for p in avg_points]
    avg_psnr = [p[1] for p in avg_points]
    
    return avg_bpp, avg_psnr


def plot_rd_curves(results_dict, output_path, dataset_name='UVG', timing_info=None, show_fps=True):
    """
    Plot RD curves for multiple configurations.
    
    Args:
        results_dict: Dictionary mapping configuration name to (bpp_list, psnr_list)
        output_path: Path to save the output figure
        dataset_name: Name of the dataset for the plot title
        timing_info: Dictionary with average encoding and decoding times (optional)
        show_fps: Whether to show FPS information (default: True)
    """
    # Use square figure
    plt.figure(figsize=(10, 10))
    
    # Define colors and markers for different configurations
    markers = ['o-', 's-', '^-', 'D-', 'v-', '*-']
    
    for idx, (config_name, (bpp_list, psnr_list)) in enumerate(results_dict.items()):
        if len(bpp_list) > 0 and len(psnr_list) > 0:
            marker_style = markers[idx % len(markers)]
            plt.plot(bpp_list, psnr_list, marker_style, 
                    label=config_name, linewidth=2, markersize=8)
            
            # Print statistics
            print(f"\n{config_name}:")
            print(f"  Number of rate points: {len(bpp_list)}")
            for i, (bpp, psnr) in enumerate(zip(bpp_list, psnr_list)):
                print(f"  Rate {i}: BPP={bpp:.6f}, PSNR={psnr:.4f} dB")
    
    plt.xlabel('BPP (bits per pixel)', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.title(f'Rate-Distortion Curves - {dataset_name}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Add timing information as text box (only in single-file mode)
    if timing_info and show_fps:
        avg_enc = timing_info.get('encoding', 0)
        avg_dec = timing_info.get('decoding', 0)
        
        # Calculate FPS (frames per second)
        enc_fps = 1.0 / avg_enc if avg_enc > 0 else 0
        dec_fps = 1.0 / avg_dec if avg_dec > 0 else 0
        
        timing_text = f'Encoding: {enc_fps:.2f} FPS\n'
        timing_text += f'Decoding: {dec_fps:.2f} FPS'
        
        # Add text box in the bottom right corner
        plt.text(0.98, 0.02, timing_text,
                transform=plt.gca().transAxes,
                fontsize=11,
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        print(f"\nTiming Statistics:")
        print(f"  Average encoding time: {avg_enc*1000:.3f} ms/frame ({avg_enc:.6f} s/frame)")
        print(f"  Encoding FPS: {enc_fps:.2f} fps")
        print(f"  Average decoding time: {avg_dec*1000:.3f} ms/frame ({avg_dec:.6f} s/frame)")
        print(f"  Decoding FPS: {dec_fps:.2f} fps")
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")
    
    plt.close()


def main():
    """Main function to analyze results and generate RD curves"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Analyze video compression results and generate RD curves.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Example usage:
  # Single file mode (with FPS display):
  python analyze_rd_curves.py results/output_UVG_RTX4090.json
  
  # Joint mode (multiple files, no FPS):
  python analyze_rd_curves.py -j results/output_UVG_RTX4090.json results/output_UVG_RTX4090_intra_period_32.json results/output_UVG_RTX4090_no_cuda_infer.json -d UVG
        '''
    )
    parser.add_argument('input_files', type=str, nargs='+',
                        help='Path(s) to the input JSON result file(s)')
    parser.add_argument('-j', '--joint', action='store_true',
                        help='Joint mode: plot multiple files on the same figure (no FPS display)')
    parser.add_argument('-d', '--dataset', type=str, default='UVG',
                        help='Dataset name to analyze (default: UVG)')
    parser.add_argument('--output-dir', type=str, default='results/curve',
                        help='Output directory for the generated curve (default: results/curve)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    # Check mode
    if not args.joint and len(args.input_files) > 1:
        print("Warning: Multiple input files provided without --joint flag.")
        print("Only the first file will be processed. Use --joint to plot all files together.")
        args.input_files = [args.input_files[0]]
    
    # Process files
    all_results = {}
    all_timing_data = []
    
    for json_file in args.input_files:
        json_path = Path(json_file)
        
        if not json_path.exists():
            print(f"Error: File not found: {json_path}")
            if args.joint:
                continue
            else:
                return
        
        print(f"Processing {json_path}...")
        
        # Load data
        data = load_json_results(json_path)
        
        # Determine dataset name
        dataset_name = args.dataset
        if dataset_name not in data:
            print(f"Warning: Dataset '{dataset_name}' not found in {json_path}")
            if not args.joint:
                # In single file mode, try to use first available dataset
                if len(data) > 0:
                    dataset_name = list(data.keys())[0]
                    print(f"Using dataset: {dataset_name}")
                else:
                    print("Error: No dataset found in JSON file")
                    return
            else:
                continue
        
        # Extract RD points and timing data
        rate_data, timing_data = extract_rd_points(data, dataset_name)
        
        if not rate_data:
            print(f"Error: No data found for dataset '{dataset_name}'")
            if args.joint:
                continue
            else:
                return
        
        # Compute averages
        avg_bpp, avg_psnr = compute_average_rd_points(rate_data)
        
        # Store timing data
        if timing_data['encoding'] and timing_data['decoding']:
            all_timing_data.append({
                'encoding': np.mean(timing_data['encoding']),
                'decoding': np.mean(timing_data['decoding'])
            })
        
        # Extract a meaningful name from the filename
        config_name = json_path.stem  # filename without extension
        all_results[config_name] = (avg_bpp, avg_psnr)
    
    if not all_results:
        print("Error: No valid data to plot!")
        return
    
    # Determine output filename and timing info
    if args.joint:
        # Joint mode: use dataset name for output filename
        output_filename = f'{dataset_name}_joint.png'
        output_path = output_dir / output_filename
        show_fps = False
        avg_timing = None
    else:
        # Single file mode: use input filename for output
        first_file = Path(args.input_files[0])
        output_filename = f'{first_file.stem}.png'
        output_path = output_dir / output_filename
        show_fps = True
        avg_timing = all_timing_data[0] if all_timing_data else None
    
    # Plot RD curves
    plot_rd_curves(all_results, output_path, dataset_name, 
                   timing_info=avg_timing, show_fps=show_fps)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == '__main__':
    main()

