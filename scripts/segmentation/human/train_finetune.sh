python3.9 train_seg.py train \
  --dataroot ./datasets/HumanBody-NS-256-3 \
	--weight_decay 0.05 --optim adamw \
	--lr 1e-4 --n_epoch 100 \
	--batch_size 16 --gamma 0.1 \
	--heads 6 --patch_size 64 --channel 10 \
	--dim 768 --encoder_depth 12 \
	--decoder_depth 6 --decoder_dim 512 --decoder_num_heads 16 \
  --augment_scale --augment_orient \
	--name "human_fine" --face_pos --lw1 1 --lw2 1 \
	--dataset_name human --seg_parts 8 \
	--checkpoint ./checkpoints/shapenet_pretrain.pkl

