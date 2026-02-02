#!/usr/bin/env python3
"""
Script pour créer un zip des données d'entraînement à uploader sur Codabench comme ressource publique.

Les données d'entraînement doivent être uploadées dans la section Resources de Codabench
et rendues publiques pour que les participants puissent les télécharger.

Usage:
    python3 create_training_data_zip.py
    python3 create_training_data_zip.py --output training_data.zip
"""

import os
import zipfile
import argparse
from pathlib import Path

def create_training_data_zip(output_name="training_data.zip", include_labels=True):
    """
    Crée un zip des données d'entraînement pour upload sur Codabench.
    
    Args:
        output_name: Nom du fichier zip de sortie
        include_labels: Si True, inclut aussi les labels d'entraînement
    """
    script_dir = Path(__file__).parent
    train_dir = script_dir / "input_data" / "train"
    output_path = script_dir / output_name
    
    if not train_dir.exists():
        raise FileNotFoundError(f"Training data directory not found: {train_dir}")
    
    # Compter les fichiers
    train_files = list(train_dir.glob("*.npz"))
    if not train_files:
        raise ValueError(f"No .npz files found in {train_dir}")
    
    # Supprimer l'ancien zip si il existe
    if output_path.exists():
        print(f"Removing old {output_name}...")
        output_path.unlink()
    
    print(f"Creating training data zip...")
    print(f"  Source: {train_dir}")
    print(f"  Files: {len(train_files)} training images")
    print(f"  Output: {output_path}")
    
    # Créer le zip
    added_count = 0
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Ajouter tous les fichiers d'entraînement
        for file_path in train_files:
            # Chemin dans le zip: train/filename.npz
            arcname = f"train/{file_path.name}"
            zipf.write(file_path, arcname=arcname)
            added_count += 1
            if added_count % 1000 == 0:
                print(f"  Added {added_count}/{len(train_files)} files...")
        
        # Ajouter les labels d'entraînement si demandé
        if include_labels:
            labels_file = script_dir / "reference_data" / "train_labels.json"
            if labels_file.exists():
                zipf.write(labels_file, arcname="train_labels.json")
                print(f"  ✓ Added train_labels.json")
            else:
                print(f"  ⚠ train_labels.json not found (optional)")
    
    print(f"\n✓ Training data zip created: {output_path}")
    print(f"  Size: {output_path.stat().st_size / (1024*1024):.1f} MB")
    print(f"  Files: {added_count} training images")
    print(f"\nTo upload on Codabench:")
    print(f"  1. Go to Resources page on Codabench")
    print(f"  2. Upload {output_name}")
    print(f"  3. Make it PUBLIC (important!)")
    print(f"  4. Update the Data page with the download link")
    print(f"\nNote: This zip contains training data only. Test data is provided")
    print(f"      automatically during submission evaluation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a zip of training data for Codabench Resources"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="training_data.zip",
        help="Output zip filename (default: training_data.zip)"
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Don't include training labels in the zip"
    )
    
    args = parser.parse_args()
    create_training_data_zip(args.output, include_labels=not args.no_labels)
