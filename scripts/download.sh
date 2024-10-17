# ModelNet40
mkdir -p data/ModelNet40
wget --no-check-certificate -P data/ModelNet40 https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip
cd data/ModelNet40
unzip modelnet40_normal_resampled.zip

# processed s3dis
wget https://huggingface.co/datasets/Pointcept/s3dis-compressed/resolve/main/s3dis.tar.gz?download=true
tar -xvf s3dis.tar.gz
