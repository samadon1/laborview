"""
LaborView AI - HuggingFace Jobs Launcher
Launch training on HF compute infrastructure
"""

# /// script
# dependencies = [
#   "huggingface_hub>=0.20.0",
# ]
# ///

import argparse
from pathlib import Path


def launch_training_job():
    """Launch training job on HuggingFace"""
    parser = argparse.ArgumentParser(description="Launch LaborView training on HF Jobs")
    parser.add_argument("--hardware", type=str, default="gpu.a10g.small",
                        choices=["gpu.t4.small", "gpu.a10g.small", "gpu.a10g.large", "gpu.a100.large"],
                        help="Hardware to use")
    parser.add_argument("--data-url", type=str, required=True,
                        help="URL to dataset (Zenodo or HF dataset)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--edge", action="store_true",
                        help="Train edge-optimized model")
    args = parser.parse_args()

    # Build the UV script command
    script_content = f'''
# Download dataset
import urllib.request
import zipfile
from pathlib import Path

data_dir = Path("./data")
data_dir.mkdir(exist_ok=True)

# Download from Zenodo
print("Downloading dataset...")
urllib.request.urlretrieve("{args.data_url}", "dataset.zip")

print("Extracting...")
with zipfile.ZipFile("dataset.zip", "r") as z:
    z.extractall(data_dir)

# Run training
import subprocess
subprocess.run([
    "python", "train.py",
    "--data-dir", str(data_dir / "DatasetV3"),
    "--epochs", "{args.epochs}",
    {"'--edge'," if args.edge else ""}
], check=True)

# Export for edge if trained edge model
if {args.edge}:
    subprocess.run([
        "python", "edge_export.py",
        "--checkpoint", "./outputs/*/best.pt",
        "--output-dir", "./edge_models",
    ], check=True)

print("Training complete!")
'''

    print(f"""
{'='*60}
LaborView AI - HuggingFace Jobs Training
{'='*60}

To launch training, use the HF Jobs CLI or API:

Option 1: Using hf-mcp tools (if available)
-------------------------------------------
Use the hf_jobs tool with operation="uv" and provide:
- script: The train.py script
- hardware: {args.hardware}

Option 2: Using HF CLI
-----------------------
huggingface-cli jobs run \\
    --hardware {args.hardware} \\
    --script train.py \\
    --requirements requirements.txt

Option 3: Manual steps
----------------------
1. Upload laborview/ folder to a HF Space or repo
2. Use HF Jobs API to run train.py

{'='*60}
""")

    return script_content


def main():
    launch_training_job()


if __name__ == "__main__":
    main()
