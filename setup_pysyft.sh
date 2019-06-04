dir="PySyft"

if [ ! -d $dir ]
then
    mkdir -p $dir
fi

conda create -p "$dir"/env python=3.6.8 anaconda

sudo yum -y groupinstall 'Development Tools'
conda activate "$dir"/env
pip install syft
