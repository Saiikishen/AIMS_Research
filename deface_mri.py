#!/usr/bin/env python3
"""
deface_mri.py

Deface one NIfTI MRI file or all NIfTI files inside a folder.

The original MRI files are not modified.

Examples
--------
Single MRI:
    python deface_mri.py /path/to/t1.nii.gz

Folder:
    python deface_mri.py /path/to/mri_folder

Specify output folder:
    python deface_mri.py /path/to/mri_folder \
        --output-dir /path/to/defaced_mri

Overwrite existing outputs:
    python deface_mri.py /path/to/mri_folder \
        --output-dir /path/to/defaced_mri \
        --overwrite
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def is_nifti_file(path: Path) -> bool:
    """Return True for .nii and .nii.gz files."""
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def remove_nifti_extension(filename: str) -> str:
    """Remove .nii or .nii.gz from a filename."""
    lower_name = filename.lower()

    if lower_name.endswith(".nii.gz"):
        return filename[:-7]

    if lower_name.endswith(".nii"):
        return filename[:-4]

    return Path(filename).stem


def find_nifti_files(input_path: Path) -> list[Path]:
    """Find one or more NIfTI files."""
    if input_path.is_file():
        if not is_nifti_file(input_path):
            raise ValueError(
                f"Input file is not a NIfTI image: {input_path}"
            )
        return [input_path]

    if input_path.is_dir():
        files = sorted(
            path
            for path in input_path.rglob("*")
            if path.is_file() and is_nifti_file(path)
        )

        if not files:
            raise FileNotFoundError(
                f"No .nii or .nii.gz files found in: {input_path}"
            )

        return files

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def check_pydeface() -> None:
    """Check whether the pydeface executable is available."""
    if shutil.which("pydeface") is None:
        raise RuntimeError(
            "pydeface was not found.\n"
            "Install it using:\n"
            "    pip install pydeface\n\n"
            "Also ensure that FSL is installed and configured."
        )


def deface_mri(
    input_file: Path,
    output_file: Path,
    overwrite: bool = False,
) -> bool:
    """
    Deface one MRI using PyDeface.

    Returns True when processing succeeds.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_file.exists() and not overwrite:
        print(f"Skipping existing file: {output_file}")
        return True

    command = [
        "pydeface",
        str(input_file),
        "--outfile",
        str(output_file),
        "--force",
    ]

    print("\nDefacing:")
    print(f"  Input : {input_file}")
    print(f"  Output: {output_file}")

    try:
        subprocess.run(
            command,
            check=True,
            text=True,
        )

        if not output_file.exists():
            print(f"ERROR: Output was not created: {output_file}")
            return False

        print("Status: completed")
        return True

    except subprocess.CalledProcessError as error:
        print(f"ERROR: PyDeface failed with code {error.returncode}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove identifiable facial structures from NIfTI MRI scans."
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input .nii/.nii.gz file or folder containing NIfTI files.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. By default, a 'defaced' folder is created "
            "beside the input file or inside the input folder."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing defaced files.",
    )

    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()

    try:
        check_pydeface()
        input_files = find_nifti_files(input_path)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"ERROR: {error}")
        sys.exit(1)

    if args.output_dir is not None:
        output_root = args.output_dir.expanduser().resolve()
    elif input_path.is_file():
        output_root = input_path.parent / "defaced"
    else:
        output_root = input_path / "defaced"

    successful = 0
    failed = 0

    for input_file in input_files:
        # Prevent reprocessing files already located in the output directory.
        try:
            input_file.relative_to(output_root)
            continue
        except ValueError:
            pass

        if input_path.is_dir():
            relative_parent = input_file.parent.relative_to(input_path)
        else:
            relative_parent = Path()

        base_name = remove_nifti_extension(input_file.name)
        output_file = (
            output_root
            / relative_parent
            / f"{base_name}_defaced.nii.gz"
        )

        if deface_mri(
            input_file=input_file,
            output_file=output_file,
            overwrite=args.overwrite,
        ):
            successful += 1
        else:
            failed += 1

    print("\nProcessing summary")
    print("------------------")
    print(f"Successful: {successful}")
    print(f"Failed    : {failed}")
    print(f"Output    : {output_root}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()