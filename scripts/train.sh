cd $(dirname $(dirname "$0")) || exit
ROOT_DIR=$(pwd)
PYTHON=python
TRAIN_CODE=train.py

DATASET=ModelNet40
CONFIG=exp0
EXP_NAME=exp0

WEIGHT=None
RESUME=true
FINETUNE=false
GPU=None

if [ "${GPU}" = 'None' ]
then
  GPU=`$PYTHON -c 'import torch; print(torch.cuda.device_count())'`
fi

echo " =========> SETTING <========="
echo "ROOT_DIR: $ROOT_DIR"
echo "Experiment name: $EXP_NAME"
echo "Dataset: $DATASET"
echo "Config: $CONFIG"
echo "GPU Num: $GPU"

EXP_DIR=exp/${DATASET}/${EXP_NAME}
MODEL_DIR=${EXP_DIR}/model
CODE_DIR=${EXP_DIR}/code
CONFIG_DIR=configs/${DATASET}/${CONFIG}.py

echo " =========> CREATE EXP DIR <========="
echo "Experiment dir: $ROOT_DIR/$EXP_DIR"
if ${RESUME}
then
  CONFIG_DIR=${EXP_DIR}/config.py
  WEIGHT=$MODEL_DIR/model_last.pth
else
  mkdir -p "$MODEL_DIR" "$CODE_DIR"
  cp -r scripts tools pointbnn "$CODE_DIR"
fi

if ${FINETUNE}
then
  CONFIG_DIR=${EXP_DIR}/config.py
  WEIGHT=$MODEL_DIR/model_best.pth
else
  mkdir -p "$MODEL_DIR" "$CODE_DIR"
  cp -r scripts tools pointbnn "$CODE_DIR"
fi

echo "Loading config in:" $CONFIG_DIR
export PYTHONPATH=./$CODE_DIR
echo "Running code in: $PYTHONPATH"

echo " =========> RUN TASK <========="
if [ "${WEIGHT}" = "None" ]
then
    $PYTHON "$CODE_DIR"/tools/$TRAIN_CODE \
    --config-file "$CONFIG_DIR" \
    --num-gpus "$GPU" \
    --options save_path="$EXP_DIR"
else
    $PYTHON "$CODE_DIR"/tools/$TRAIN_CODE \
    --config-file "$CONFIG_DIR" \
    --num-gpus "$GPU" \
    --options save_path="$EXP_DIR" resume="$RESUME" weight="$WEIGHT" finetune="$FINETUNE"
fi