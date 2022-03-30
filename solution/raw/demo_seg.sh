conda activate video_understand
export PYTHONPATH=/home/haoren/repo/slowfast/:$PYTHONPATH
cd /home/haoren/repo/slowfast/
python tools/run_net.py \
  --cfg configs/ChinaPost/SLOWFAST_SEG_8x8_R50.yaml \
  TEST.CHECKPOINT_FILE_PATH /home/haoren/repo/slowfast/checkpoints/slowfast_seg_channel64_beta4_class2_pure_action_add0315_balance_pos_neg_balance_scenario_full_equal_lr0p0001/checkpoints/checkpoint_epoch_29800.pyth \
  NUM_GPUS 1 \
  TRAIN.ENABLE False \
  TEST.ENABLE False \