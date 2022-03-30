conda activate video_understand
export PYTHONPATH=/home/haoren/repo/slowfast/:$PYTHONPATH
cd /home/haoren/repo/slowfast/
python tools/run_net.py \
  --cfg configs/ChinaPost/SLOWFAST_8x8_R50.yaml \
  DATA.PATH_TO_DATA_DIR /home/haoren/repo/slowfast/data/raw/anno_time_clip/ \
  DATA.PATH_PREFIX /home/haoren/repo/slowfast/data/raw/ \
  NUM_GPUS 1 \
  TRAIN.BATCH_SIZE 4 \
  TEST.ENABLE False \
  DEMO.ENABLE False \