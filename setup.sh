cd ..
git clone --depth 1 https://github.com/camenduru/simple-knn.git
git clone --depth 1 https://github.com/rahul-goel/fused-ssim.git

pip install ninja plyfile tqdm

pip3 install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124

cd simple-knn
python setup.py install

cd ../fused-ssim
python setup.py install

cd ../fast-gs/submodules/diff-gaussian-rasterization_fastgs
python setup.py install
