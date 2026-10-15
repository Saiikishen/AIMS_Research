#!/usr/bin/env python3

import subprocess
from pathlib import Path

# ====================================================
# CHANGE THIS PATH
# ====================================================
MAIN_FOLDER = Path("/Volumes/PORTABLESSD/PASCOM for Nectara/30 Patient Data to Nectara")
# ====================================================


def find_image(patient_folder: Path, base_name: str):
    """
    Find either:
      base_name.nii
      base_name.nii.gz
    """

    nii_file = patient_folder / f"{base_name}.nii"
    niigz_file = patient_folder / f"{base_name}.nii.gz"

    if nii_file.exists():
        return nii_file

    if niigz_file.exists():
        return niigz_file

    return None


def deface_image(input_file: Path, output_file: Path):
    """
    Deface one image independently using PyDeface.

    The original image is preserved.
    The output is saved as an uncompressed .nii file.
    """

    if output_file.exists():
        print(f"  Skipping existing output: {output_file.name}")
        return True

    command = [
        "pydeface",
        str(input_file),
        "--outfile",
        str(output_file),
        "--force",
    ]

    print(f"  Input : {input_file.name}")
    print(f"  Output: {output_file.name}")

    try:
        subprocess.run(command, check=True)

        if output_file.exists():
            print("  Status: completed")
            return True

        print("  Status: output was not created")
        return False

    except subprocess.CalledProcessError as error:
        print(f"  Status: failed, return code {error.returncode}")
        return False

    except FileNotFoundError:
        print("  Status: pydeface command was not found")
        print("  Install it using: pip install pydeface")
        return False


def main():
    if not MAIN_FOLDER.exists():
        raise FileNotFoundError(
            f"Main folder does not exist: {MAIN_FOLDER}"
        )

    if not MAIN_FOLDER.is_dir():
        raise NotADirectoryError(
            f"MAIN_FOLDER is not a folder: {MAIN_FOLDER}"
        )

    patient_folders = sorted(
        folder
        for folder in MAIN_FOLDER.iterdir()
        if folder.is_dir()
    )

    if not patient_folders:
        print(f"No patient folders found inside: {MAIN_FOLDER}")
        return

    # Images will be defaced independently because they are not coregistered.
    modalities = ["t1", "flair", "pet"]

    total_success = 0
    total_failed = 0
    total_missing = 0
    total_skipped = 0

    for patient_folder in patient_folders:
        print("\n" + "=" * 60)
        print(f"Processing patient: {patient_folder.name}")
        print("=" * 60)

        for modality in modalities:
            image_file = find_image(patient_folder, modality)

            if image_file is None:
                print(f"  {modality.upper()} image not found")
                total_missing += 1
                continue

            output_file = patient_folder / f"{modality}_defaced.nii"

            if output_file.exists():
                print(f"  Skipping existing output: {output_file.name}")
                total_skipped += 1
                continue

            success = deface_image(
                input_file=image_file,
                output_file=output_file,
            )

            if success:
                total_success += 1
            else:
                total_failed += 1

    print("\n" + "=" * 60)
    print("Processing summary")
    print("=" * 60)
    print(f"Successfully defaced : {total_success}")
    print(f"Already existed      : {total_skipped}")
    print(f"Images not found     : {total_missing}")
    print(f"Failed               : {total_failed}")
    print(f"Main folder          : {MAIN_FOLDER}")


if __name__ == "__main__":
    main()