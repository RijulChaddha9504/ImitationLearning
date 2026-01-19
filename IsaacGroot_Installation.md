# Isaac GR00T Installation Instructions

Run these commands inside the WatCloud Docker container:

```bash
cd ~
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
cd ~/Isaac-GR00T
sed -i 's/"flash-attn[^"]*",//' pyproject.toml
uv sync --python 3.10
uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
source .venv/bin/activate
python -c 'import gr00t; print("GR00T ready!")'
```
