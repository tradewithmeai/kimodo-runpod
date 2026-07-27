set -e
PIP=./venv/Scripts/pip.exe
PY=./venv/Scripts/python.exe
echo "=== [2/4] deps ==="
$PIP install -q --no-input numpy scipy einops smplx blobfile ftfy regex fastapi "uvicorn[standard]" gdown
$PIP install -q --no-input git+https://github.com/openai/CLIP.git
echo "deps ok"
echo "=== [3/4] checkpoint ==="
mkdir -p motion-diffusion-model/save && cd motion-diffusion-model/save
if [ ! -f humanml_enc_512_50steps/model000750000.pt ]; then
  ../../venv/Scripts/gdown.exe 1cfadR1eZ116TIdXK7qDX1RugAerEiJXr 2>&1 | tail -1
  ../../venv/Scripts/python.exe -c "import zipfile; zipfile.ZipFile('humanml_enc_512_50steps.zip').extractall('.')"
  rm -f humanml_enc_512_50steps.zip
fi
ls humanml_enc_512_50steps/ && cd ../..
echo "=== [4/4] smoke test ==="
export MDM_REPO="D:/Documents/11Projects/Kimodo/local/motion-diffusion-model"
$PY -W ignore smoke_test.py
echo "INSTALL_COMPLETE"
