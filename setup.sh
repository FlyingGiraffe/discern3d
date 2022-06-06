# run this in your conda environment!
conda create --name discern3d python=3.7
conda activate discern3d
conda install pytorch cudatoolkit=10.2 -c pytorch
conda install pyg -c pyg
pip install -r requirements.txt