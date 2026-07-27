set -e
PIP=./venv/Scripts/pip.exe
PY=./venv/Scripts/python.exe
echo "=== [1/4] torch cu126 (large download) ==="
$PIP install -q --no-input torch --index-url https://download.pytorch.org/whl/cu126
echo "=== [2/4] deps ==="
$PIP install -q --no-input numpy scipy einops smplx blobfile ftfy regex fastapi "uvicorn[standard]" gdown
$PIP install -q --no-input git+https://github.com/openai/CLIP.git
echo "=== [3/4] checkpoint ==="
mkdir -p motion-diffusion-model/save && cd motion-diffusion-model/save
../../venv/Scripts/gdown.exe 1cfadR1eZ116TIdXK7qDX1RugAerEiJXr 2>&1 | tail -1
$PY -c "import zipfile; zipfile.ZipFile('humanml_enc_512_50steps.zip').extractall('.')"
rm -f humanml_enc_512_50steps.zip && cd ../..
echo "=== [4/4] smoke test ==="
export MDM_REPO="D:/Documents/11Projects/Kimodo/local/motion-diffusion-model"
$PY -W ignore smoke_test.py
echo "INSTALL_COMPLETE"
