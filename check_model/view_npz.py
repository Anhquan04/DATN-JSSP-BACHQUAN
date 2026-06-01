
import sys
import numpy as np
import argparse
from pathlib import Path


def view_npz(npz_path: str, limit: int = None, save_file: str = None):
    """Load .npz và in nội dung chi tiết."""
    
    npz_path = Path(npz_path)
    if not npz_path.exists():
        print(f"❌ File not found: {npz_path}")
        return
    
    print(f"\n{'='*80}")
    print(f"VIEWING: {npz_path.name}")
    print(f"{'='*80}\n")
    
    try:
        data = np.load(str(npz_path))
    except Exception as e:
        print(f"❌ Failed to load: {e}")
        return
    
    # In danh sách tất cả keys
    print("📋 ALL KEYS IN FILE:")
    print("-" * 80)
    for i, key in enumerate(sorted(data.files), 1):
        arr = data[key]
        if isinstance(arr, np.ndarray):
            print(f"{i:2d}. {key:20s} | shape={str(arr.shape):20s} | dtype={arr.dtype}")
        else:
            print(f"{i:2d}. {key:20s} | scalar value")
    
    print("\n" + "="*80)
    print("CONTENT DETAILS:")
    print("="*80 + "\n")
    
    # In nội dung chi tiết từng key
    output_lines = []
    
    for key in sorted(data.files):
        arr = data[key]
        
        output_lines.append(f"\n{'='*80}")
        output_lines.append(f"KEY: {key}")
        output_lines.append(f"{'='*80}")
        
        if isinstance(arr, np.ndarray):
            output_lines.append(f"Shape:    {arr.shape}")
            output_lines.append(f"Dtype:    {arr.dtype}")
            output_lines.append(f"Size:     {arr.size} elements")
            output_lines.append(f"Min:      {np.min(arr)}")
            output_lines.append(f"Max:      {np.max(arr)}")
            output_lines.append(f"Mean:     {np.mean(arr)}")
            output_lines.append(f"Std:      {np.std(arr)}")
            output_lines.append("")
            output_lines.append("CONTENT:")
            output_lines.append("-" * 80)
            
            # Flatten để in tiện
            flat_arr = arr.flatten()
            
            if limit and flat_arr.size > limit:
                # In limit phần tử đầu + cuối
                output_lines.append(f"[Showing first {limit} of {flat_arr.size} elements]")
                output_lines.append("")
                for i in range(limit):
                    output_lines.append(f"  [{i}] = {flat_arr[i]}")
                output_lines.append("")
                output_lines.append("  ...")
                output_lines.append("")
                for i in range(max(limit, flat_arr.size-limit), flat_arr.size):
                    output_lines.append(f"  [{i}] = {flat_arr[i]}")
            else:
                # In tất cả
                for i, val in enumerate(flat_arr):
                    output_lines.append(f"  [{i}] = {val}")
        else:
            # Scalar
            output_lines.append(f"Value: {arr}")
    
    # In ra console + save nếu cần
    full_output = "\n".join(output_lines)
    print(full_output)
    
    if save_file:
        with open(save_file, "w", encoding="utf-8") as f:
            f.write(f"NPZ FILE: {npz_path.name}\n")
            f.write(full_output)
        print(f"\n✅ Saved to: {save_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View .npz file content")
    parser.add_argument("npz_path", type=str, help="Path to .npz file")
    parser.add_argument("--limit", type=int, default=10,
                        help="Max elements to show per array (default: 10)")
    parser.add_argument("--save", type=str, help="Save output to file")
    parser.add_argument("--all", action="store_true",
                        help="Show all elements (no limit)")
    
    args = parser.parse_args()
    
    limit = None if args.all else (args.limit if args.limit else 10)
    
    view_npz(args.npz_path, limit=limit, save_file=args.save)