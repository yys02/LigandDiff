# LigandDiff: 3D Transition Metal Complex Ligand Generation

[multi-LigandDiff](https://github.com/Neon8988/multi_LigandDiff) is now available.

[LigandDiff](https://pubs.acs.org/doi/full/10.1021/acs.jctc.4c00232) - This repository is the official implementation of the ligands generation for 3D transition metal complexes.

![diffusion models](https://github.com/Neon8988/LigandDiff/blob/main/image/example.gif)

## Quick Start

### 1. Environment Setup

The easiest way to set up the environment is using the provided conda environment:

```bash
# Clone the repository
git clone <repository-url>
cd LigandDiff

# For Ubuntu 22.04 + CUDA 12.4 (recommended)
./setup_ubuntu2204.sh
conda activate liganddiff

# Or use the general setup script
./setup_environment.sh
conda activate liganddiff

# Or manual setup
conda env create -f environment.yml
conda activate liganddiff
```

For detailed setup instructions, see [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md).

### 2. Dataset

Download all datasets from this [link](https://zenodo.org/records/10651292)

### 3. Training

```bash
# Train the model
python train.py --config config.yml
```

### 4. Generation

To generate ligands, run generate.py with your own complex in .xyz format:

```bash
python generate.py --model path/to/model.ckpt --complex path/to/complex.xyz --outdir output/
```

## Environment Files

- `environment.yml` - Conda environment configuration with all dependencies
- `setup_environment.sh` - Automated setup script
- `test_environment.py` - Dependency validation script
- `ENVIRONMENT_SETUP.md` - Detailed setup guide

## System Requirements

- **Python**: 3.9+
- **RAM**: 8GB minimum, 16GB+ recommended
- **GPU**: Optional but recommended for faster training
- **Storage**: 10GB+ free space

## Citation

If you use this code in your research, please cite:

```bibtex
@article{liganddiff2024,
  title={LigandDiff: A Diffusion Model for 3D Transition Metal Complex Ligand Generation},
  author={...},
  journal={Journal of Chemical Theory and Computation},
  year={2024}
}
```

## License

See [LICENSE](LICENSE) file for details. 
