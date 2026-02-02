#!/usr/bin/env python3
"""
Script pour créer un zip de soumission correct pour Codabench.
Ce script crée un zip avec model.py et requirements.txt à la racine.

Usage:
    python3 create_submission_zip.py
    python3 create_submission_zip.py --model path/to/model.py
"""

import os
import sys
import zipfile
import argparse
from pathlib import Path

def create_submission_zip(model_path=None, output_name="submission.zip", use_tensorflow=False):
    """
    Crée un zip de soumission avec model.py et requirements.txt à la racine.
    
    Args:
        model_path: Chemin vers model.py (par défaut: sample_code_submission/model.py)
        output_name: Nom du fichier zip de sortie
        use_tensorflow: Si True, utilise model_tensorflow.py (TensorFlow version) au lieu de model.py (scikit-learn baseline)
    """
    script_dir = Path(__file__).parent
    
    # Déterminer le chemin du modèle
    if model_path is None:
        if use_tensorflow:
            model_path = script_dir / "sample_code_submission" / "model_tensorflow.py"
        else:
            # Par défaut: baseline scikit-learn (model.py)
            model_path = script_dir / "sample_code_submission" / "model.py"
    else:
        model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Chemin vers requirements.txt selon le type de modèle
    if use_tensorflow:
        requirements_path = model_path.parent / "requirements_tensorflow.txt"
    else:
        requirements_path = model_path.parent / "requirements.txt"
    
    # Chemin de sortie
    output_path = script_dir / output_name
    
    # Supprimer l'ancien zip si il existe
    if output_path.exists():
        print(f"Removing old {output_name}...")
        output_path.unlink()
    
    print(f"Creating submission zip: {output_name}")
    print(f"  - Model: {model_path.name}")
    
    # Créer le zip
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Ajouter model.py à la racine du zip (toujours nommé model.py dans le zip)
        zipf.write(model_path, arcname="model.py")
        print(f"  ✓ Added {model_path.name} as model.py")
        
        # Ajouter requirements.txt si il existe
        if requirements_path.exists():
            zipf.write(requirements_path, arcname="requirements.txt")
            print(f"  ✓ Added {requirements_path.name} as requirements.txt")
        else:
            print(f"  ⚠ requirements.txt not found (optional)")
    
    print(f"\n✓ Submission zip created: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024:.1f} KB")
    print(f"\nTo submit:")
    print(f"  1. Upload {output_name} to Codabench")
    print(f"  2. Make sure model.py and requirements.txt are at the root of the zip")
    
    # Vérifier le contenu du zip
    print(f"\nVerifying zip contents:")
    with zipfile.ZipFile(output_path, 'r') as zipf:
        files = zipf.namelist()
        for file in files:
            print(f"  - {file}")
        
        if "model.py" not in files:
            print("\n⚠ WARNING: model.py not found in zip!")
        if "requirements.txt" not in files:
            print("⚠ WARNING: requirements.txt not found in zip (optional but recommended)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a submission zip for Codabench"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model.py file (default: sample_code_submission/model.py)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="submission.zip",
        help="Output zip filename (default: submission.zip)"
    )
    parser.add_argument(
        "--tensorflow",
        action="store_true",
        help="Use TensorFlow model (model_tensorflow.py) instead of default scikit-learn baseline (model.py)"
    )
    
    args = parser.parse_args()
    
    create_submission_zip(args.model, args.output, args.tensorflow)
