dir="PySyft"

if [ ! -d $dir ]
then
    mkdir -p $dir
fi

cd $dir
conda create -p "$dir"/env python=3.7 anaconda

sudo yum -y groupinstall 'Development Tools'
conda activate -p ./env
pip install syft
