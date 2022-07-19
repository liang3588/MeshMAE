#finetune shapenet
# Train
python3.9 train_cls.py train \
	--dataroot ./datasets/Manifold40-MAPS-96-3/ \
	--batch_size 32 --augment_scale --n_classes 40 \
	--channels 10 --patch_size 64 --n_epoch 100 \
	--name "manifoldBase_fine_sn" \
	--weight_decay 0.05 \
	--lr 1e-4 --optim "adamw" \
	--depth 12 \
	--heads 12 \
	--lr_milestones "none" \
	--encoder_depth 12 \
	--decoder_depth 6 \
	--decoder_dim 512 \
	--decoder_num_heads 16 \
	--checkpoint "./checkpoints/shapenet_pretrain.pkl" \
	--num_warmup_steps "2" \
	--dim 768
