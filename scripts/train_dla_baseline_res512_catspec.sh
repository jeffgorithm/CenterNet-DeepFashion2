python main.py ctdet \
--dataset deepfashion2 \
--exp_id deepfashion2_dla_res512_baseline_catspec \
--gpus 0,2 \
--arch dla_34 \
--input_res 512 \
--num_epochs 50 \
--lr_step 30,40 \
--batch_size 40 \
--val_intervals 5 \
--num_workers 8 \
--neptune \
--neptune-path ../neptune.txt \
--cat_spec_wh \
--resume