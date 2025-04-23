#!/usr/bin/env python3
"""
collect_code.py – concatenate code files into a single text file

Walks the current directory and all subdirectories, finds files with common
code extensions, and writes their contents (with headers) into one output file.

Usage:
  python collect_code.py [--output ALL_CODE.txt] [--extensions .py,.js,...]
"""

import os
import argparse
from pathlib import Path

# Default extensions to include
DEFAULT_EXTS = [
    '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.hpp', '.sh', '.bat', '.yaml', '.yml', '.json', '.md'
]


def collect_code_files(root: Path, extensions: list[str]) -> list[Path]:
    """Recursively collect files under root matching the given extensions."""
    matches = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if any(fname.endswith(ext) for ext in extensions):
                matches.append(Path(dirpath) / fname)
    return matches


def concatenate_files(files: list[Path], output: Path) -> None:
    """Write each file's path and contents into the output file."""
    with output.open('w', encoding='utf-8') as out_f:
        for fpath in sorted(files):
            out_f.write(f"#### {fpath}\n")
            try:
                text = fpath.read_text(encoding='utf-8')
            except Exception:
                # Binary or unreadable file
                out_f.write(f"[Could not read contents of {fpath}]\n\n")
                continue
            out_f.write(text)
            out_f.write("\n\n")
    print(f"Collected {len(files)} files into {output}")


def parse_args():
    p = argparse.ArgumentParser(description="Concatenate code files into one text file.")
    p.add_argument(
        '--output', '-o', type=Path, default=Path('ALL_CODE.txt'),
        help='Output file path'
    )
    p.add_argument(
        '--extensions', '-e', type=lambda s: s.split(','),
        default=DEFAULT_EXTS,
        help='Comma-separated list of file extensions to include'
    )
    p.add_argument(
        '--root', '-r', type=Path, default=Path('.'),
        help='Root directory to search'
    )
    return p.parse_args()


def main():
    args = parse_args()
    files = collect_code_files(args.root, args.extensions)
    concatenate_files(files, args.output)


if __name__ == '__main__':
    main()
