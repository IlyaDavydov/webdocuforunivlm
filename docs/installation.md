# Requirements & Installation

- We strongly recommend Conda for virtual environments. Refer to the [Conda Installation Guide](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html).

```bash
conda create -n univlm python=3.10
```
Change "univlm" to any name of your choice.

## Installation

Follow the simple two-step installation process:

1. Use pip to install the library.
```bash
pip install univlm
```
2. Run the one-time backbone setup command.
```bash
univlm-install
```

### Notes
- Requires an internet connection for model downloads.
- Conda and Git must be pre-installed.
- Some methods use parallel processing for improved performance.

