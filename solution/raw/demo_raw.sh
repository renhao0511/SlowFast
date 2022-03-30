conda activate video_understand
export PYTHONPATH=/home/haoren/repo/slowfast/:$PYTHONPATH
cd /home/haoren/repo/slowfast/
python tools/run_net.py \
  --cfg configs/ChinaPost/SLOWFAST_8x8_R50.yaml \
  TEST.CHECKPOINT_FILE_PATH /home/haoren/repo/slowfast/checkpoints/raw_time_clip/checkpoints/checkpoint_epoch_00196.pyth \
  NUM_GPUS 1 \
  TRAIN.ENABLE False \
  TEST.ENABLE False \