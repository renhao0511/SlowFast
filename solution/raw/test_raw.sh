conda activate video_understand
export PYTHONPATH=/home/haoren/repo/slowfast/:$PYTHONPATH
cd /home/haoren/repo/slowfast/
python tools/run_net.py \
  --cfg configs/ChinaPost/SLOWFAST_8x8_R50.yaml \
  DATA.PATH_TO_DATA_DIR /home/haoren/repo/slowfast/data/raw/anno/ \
  DATA.PATH_PREFIX /home/haoren/repo/slowfast/data/raw/ \
  TEST.CHECKPOINT_FILE_PATH /home/haoren/repo/slowfast/checkpoints/raw/checkpoints/checkpoint_epoch_00160.pyth \
  TRAIN.ENABLE False \
  NUM_GPUS 1 \
  TRAIN.BATCH_SIZE 4 \