python generate_shop_gallery.py ctdet \
--dataset deepfashion2 \
--load_model ../exp/ctdet/deepfashion2_dla_res512_baseline/model_last.pth \
--input_res 512 \
--gpus 3 \
--vis_thresh 0.3 \
--shop_gallery