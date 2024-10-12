cd $(dirname $(dirname "$0")) || exit
ROOT_DIR=$(pwd)
PYTHON=python3
VIS_CODE=vis.py

DATASET=S3DIS
CONFIG=semseg_ptbnn_small_dynamic_batch
EXP_NAME=semseg_ptbnn_small_dynamic_batch

echo "Experiment name: $EXP_NAME"
echo "Dataset: $DATASET"

EXP_DIR=exp/${DATASET}/${EXP_NAME}
CODE_DIR=${EXP_DIR}/code
CONFIG_DIR=${EXP_DIR}/config.py

if [ "${CONFIG}" = "None" ]
then
    CONFIG_DIR=${EXP_DIR}/config.py
else
    CONFIG_DIR=configs/${DATASET}/${CONFIG}.py
fi

cp -r scripts tools pointbnn "$CODE_DIR"

echo "Loading config in:" $CONFIG_DIR
export PYTHONPATH=./$CODE_DIR
echo "Running code in: $CODE_DIR"


echo " =========> RUN TASK <========="

python3 "$CODE_DIR"/tools/$VIS_CODE \
  --config-file "$CONFIG_DIR" \
  --options save_path="$EXP_DIR"