python3 train.py \
    --train_set /data/linus/data/deep_fashion_2.csv \
    --image_root /data/linus/data/images/ \
    --experiment_root /data/linus/triplet-reid/final \
    --embedding_dim 512 \
    --batch_k 4 \
    --batch_p 3 \
    --net_input_height 512 \
    --net_input_width 512 \
    --train_iterations 100000 \
    --detailed_logs
