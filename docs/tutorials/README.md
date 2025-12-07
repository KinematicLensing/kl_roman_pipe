# Tutorials

Interactive tutorials for learning the kinematic lensing pipeline.

## Available Tutorials

- **`quickstart.md`** - Introduction to basic pipeline functionality: models, likelihoods, and optimization
- **`tng50_data.md`** - Working with TNG50 mock observations downloaded from CyVerse

## Converting to Jupyter Notebooks

All tutorials are written in Jupytext-compatible markdown format. To convert them to executable Jupyter notebooks:

```bash
# Convert all tutorials at once
make tutorials

# Or convert a specific tutorial manually
jupytext --to ipynb docs/tutorials/quickstart.md
jupytext --to ipynb docs/tutorials/tng50_data.md
```

This creates `.ipynb` files alongside the markdown files that you can open in Jupyter Lab/Notebook or VS Code.

## Requirements

To run the tutorials, you need:

1. The `klpipe` conda environment installed:
   ```bash
   make install
   ```

2. For TNG50 tutorials, download the mock data:
   ```bash
   make download-cyverse-data
   ```

## Running Tutorials

### In Jupyter
```bash
conda activate klpipe
jupyter lab
# Open the .ipynb file in your browser
```

### In VS Code
- Open the `.ipynb` file directly
- VS Code will prompt to select the `klpipe` kernel

### As Python Scripts
The markdown files are also valid Python scripts (using `# %%` cell markers):
```bash
conda run -n klpipe python docs/tutorials/quickstart.md
```
