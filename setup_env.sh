# install the lib
pip install -r requirements.txt
# install requirements for the NDS search spaces (ResNet/ResNeXt)
pip install -r instructions/nds/requirements.txt

# for zeroshot
# To run zeroshot evaluation, one should run `export PYTHONPATH=$(readlink -f ../zero-cost-nas/):$PYTHONPATH`. And check `python -c 'import foresight'` works.
export PYTHONPATH=$(readlink -f ./zero-cost-nas/):$PYTHONPATH
python -c 'import foresight'


echo "--- env setuped ---"
