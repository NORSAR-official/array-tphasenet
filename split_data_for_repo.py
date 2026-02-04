#!/usr/bin/env python
"""
Utility script to split and merge HDF5 files for repository storage.

Usage
-----
Split a large HDF5 file into smaller chunks:
    python split_data_for_repo.py split --input data.hdf5 --output-dir ./splits --chunk-size 20000

Merge split HDF5 files back into a single file:
    python split_data_for_repo.py merge --pattern "splits/data_*.hdf5" --output merged.hdf5
"""
import h5py
import numpy as np
import os
import glob
import argparse


def split_hdf5_file(
    input_file: str,
    output_dir: str,
    output_prefix: str = None,
    chunk_size: int = 20000,
    compression: str = "gzip",
    dataset_keys=None,
):
    """
    Split a large HDF5 file into smaller chunks along axis 0.

    Parameters
    ----------
    input_file : str
        Path to the input HDF5 file to split
    output_dir : str
        Directory where split files will be saved
    output_prefix : str, optional
        Prefix for output split files. If None, uses the original filename (without extension).
    chunk_size : int, optional
        Number of rows per split file (default: 20000)
    compression : str, optional
        Compression type for output files, e.g. "gzip" or None (default: "gzip")
    dataset_keys : list[str] | None, optional
        List of dataset keys to split. If None, all datasets are split.

    Returns
    -------
    list[str]
        List of created split file paths
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Use original filename (without extension) as prefix if not specified
    if output_prefix is None:
        base_name = os.path.basename(input_file)
        output_prefix = os.path.splitext(base_name)[0]
    
    output_files = []
    
    with h5py.File(input_file, "r") as fin:
        # Determine which keys to process
        if dataset_keys is None:
            dataset_keys = list(fin.keys())
        
        # Validate all keys exist
        for k in dataset_keys:
            if k not in fin:
                raise KeyError(f"Dataset '{k}' not found in {input_file}")
        
        # Get number of rows (assuming all datasets have same first dimension)
        n_rows = fin[dataset_keys[0]].shape[0]
        print(f"Splitting {input_file} with {n_rows} rows into chunks of {chunk_size}")
        print(f"Output prefix: {output_prefix}")
        
        # Split into chunks
        for i, start in enumerate(range(0, n_rows, chunk_size)):
            end = min(start + chunk_size, n_rows)
            out_file = os.path.join(output_dir, f"{output_prefix}_{i:04d}.hdf5")
            
            with h5py.File(out_file, "w") as fout:
                for k in dataset_keys:
                    dset = fin[k]
                    fout.create_dataset(
                        k,
                        data=dset[start:end],
                        dtype=dset.dtype,
                        compression=compression,
                    )
            
            output_files.append(out_file)
            print(f"Wrote {out_file}: rows {start}:{end} ({end - start} samples)")
    
    print(f"✅ Split complete: {len(output_files)} files created in {output_dir}")
    return output_files


def merge_hdf5_splits(
    split_glob: str,
    output_file: str,
    dataset_keys=None,
    compression="gzip",
    shuffle=True,
):
    """
    Merge multiple HDF5 split files into a single HDF5 file by concatenating along axis 0.

    Parameters
    ----------
    split_glob : str
        Glob pattern for split files, e.g. "split_*.hdf5"
    output_file : str
        Path for merged output file, e.g. "merged.hdf5"
    dataset_keys : list[str] | None
        Datasets to merge. If None, inferred from the first file's keys.
    compression : str | None
        e.g. "gzip" or None
    shuffle : bool
        HDF5 shuffle filter (often helps compression a bit).
    """
    files = sorted(glob.glob(split_glob))
    if not files:
        raise FileNotFoundError(f"No files match pattern: {split_glob}")

    print(f"Found {len(files)} files to merge")

    # Inspect first file to infer keys + shapes/dtypes
    with h5py.File(files[0], "r") as f0:
        if dataset_keys is None:
            dataset_keys = list(f0.keys())
        meta = {}
        for k in dataset_keys:
            if k not in f0:
                raise KeyError(f"Dataset '{k}' not found in {files[0]}")
            d = f0[k]
            if d.ndim == 0:
                raise ValueError(f"Dataset '{k}' is scalar in {files[0]} (expected row-wise array).")
            meta[k] = {
                "dtype": d.dtype,
                "shape_tail": d.shape[1:],   # everything except row dimension
            }

    # Compute total rows and validate consistency
    total_rows = 0
    for fp in files:
        with h5py.File(fp, "r") as f:
            n = None
            for k in dataset_keys:
                if k not in f:
                    raise KeyError(f"Dataset '{k}' not found in {fp}")
                d = f[k]
                if d.dtype != meta[k]["dtype"]:
                    raise TypeError(f"Dtype mismatch for '{k}' in {fp}: {d.dtype} vs {meta[k]['dtype']}")
                if d.shape[1:] != meta[k]["shape_tail"]:
                    raise ValueError(
                        f"Shape mismatch for '{k}' in {fp}: {d.shape[1:]} vs {meta[k]['shape_tail']}"
                    )
                n_k = d.shape[0]
                n = n_k if n is None else n
                if n_k != n:
                    raise ValueError(f"Row count mismatch within file {fp} between datasets.")
            total_rows += n

    # Create output datasets and stream-copy data
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with h5py.File(output_file, "w") as fout:
        out = {}
        for k in dataset_keys:
            shape = (total_rows,) + meta[k]["shape_tail"]
            out[k] = fout.create_dataset(
                k,
                shape=shape,
                dtype=meta[k]["dtype"],
                chunks=True,
                compression=compression,
                shuffle=shuffle,
            )

        write_pos = 0
        for fp in files:
            with h5py.File(fp, "r") as fin:
                n = fin[dataset_keys[0]].shape[0]
                for k in dataset_keys:
                    out[k][write_pos:write_pos + n] = fin[k][:]
            write_pos += n
            print(f"Merged {fp} ({n} rows). Progress: {write_pos}/{total_rows}")

    print(f"✅ Wrote merged file: {output_file} (rows={total_rows})")


def main():
    """Main command-line interface."""
    parser = argparse.ArgumentParser(
        description="Split or merge HDF5 files for repository storage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Split a large file into 20k-row chunks (uses original filename as prefix)
  python split_data_for_repo.py split --input data.hdf5 --output-dir ./splits --chunk-size 20000
  # Creates: splits/data_0000.hdf5, splits/data_0001.hdf5, ...
  
  # Split with custom prefix
  python split_data_for_repo.py split --input data.hdf5 --output-dir ./splits --prefix custom --chunk-size 20000
  # Creates: splits/custom_0000.hdf5, splits/custom_0001.hdf5, ...
  
  # Merge split files back together
  python split_data_for_repo.py merge --pattern "splits/data_*.hdf5" --output merged.hdf5
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Split command
    split_parser = subparsers.add_parser("split", help="Split an HDF5 file into smaller chunks")
    split_parser.add_argument("--input", required=True, help="Input HDF5 file path")
    split_parser.add_argument("--output-dir", required=True, help="Output directory for split files")
    split_parser.add_argument("--prefix", default=None, help="Prefix for split files (default: uses original filename)")
    split_parser.add_argument("--chunk-size", type=int, default=20000, help="Rows per split file (default: 20000)")
    split_parser.add_argument("--compression", default="gzip", help="Compression type (default: gzip)")
    
    # Merge command
    merge_parser = subparsers.add_parser("merge", help="Merge split HDF5 files into one")
    merge_parser.add_argument("--pattern", required=True, help="Glob pattern for split files (e.g., 'split_*.hdf5')")
    merge_parser.add_argument("--output", required=True, help="Output merged HDF5 file path")
    merge_parser.add_argument("--compression", default="gzip", help="Compression type (default: gzip)")
    merge_parser.add_argument("--no-shuffle", action="store_true", help="Disable HDF5 shuffle filter")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == "split":
        split_hdf5_file(
            input_file=args.input,
            output_dir=args.output_dir,
            output_prefix=args.prefix,
            chunk_size=args.chunk_size,
            compression=args.compression if args.compression != "none" else None,
        )
    
    elif args.command == "merge":
        merge_hdf5_splits(
            split_glob=args.pattern,
            output_file=args.output,
            compression=args.compression if args.compression != "none" else None,
            shuffle=not args.no_shuffle,
        )


if __name__ == "__main__":
    main()
