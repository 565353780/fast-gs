cd ..
git clone git@github.com:565353780/base-trainer.git
git clone --depth 1 https://github.com/camenduru/simple-knn.git
git clone --depth 1 https://github.com/rahul-goel/fused-ssim.git

pip install ninja plyfile

cd base-trainer
./dev_setup.sh

cd ../simple-knn
python setup.py install

cd ../fused-ssim
python setup.py install

cd ../fast-gs/fast_gs/Lib/diff-gaussian-rasterization_fastgs
python setup.py install
