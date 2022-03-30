# export PATH="/home/haoren/anaconda3/bin:$PATH"
# source activate /home/haoren/.conda/envs/video_understand
export PYTHONPATH=/home/haoren/repo/slowfast/:$PYTHONPATH
cd /home/haoren/repo/slowfast/
python tools/run_net.py \
  --cfg configs/ChinaPost/SLOWFAST_SEG_8x8_R50.yaml \
  DATA.PATH_TO_DATA_DIR /home/haoren/repo/slowfast/data/raw/anno_seg_pure_action_add0211_balance_pos_neg/ \
  DATA.PATH_PREFIX /home/haoren/repo/slowfast/data/raw/ \
  NUM_GPUS 1 \
  TRAIN.BATCH_SIZE 1 \
  TEST.ENABLE False \
  DEMO.ENABLE False \
  OUTPUT_DIR ./checkpoints/slowfast_seg_channel64_beta4_class2_pure_action_add0211_balance_pos_neg_lr0p0001_debug/ \