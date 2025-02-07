# Requirements & Installation

- We strongly recommend conda for virtual environment. Refer to the [Conda Installation guide](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html)

```bash
conda create -n univlm python=3.1
```
Chnage "univlm" to any name of your choice.

## Installation

Follow the simple 2 step installtion process:

1. Using pip to install library.
```bash
pip install univlm
```
2. One time backbone setup command.
```bash
univlm-install
```

### Notes
- Requires an internet connection for model downloads.
- Conda and Git must be pre-installed.
- Some methods use parallel processing for improved performance.
