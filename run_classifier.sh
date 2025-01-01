# 此脚本用于训练模型，主要用于微调阶段，但目前是直接训练。
# conda run -n "ET-Bert" python
timestamp=$(date +"%Y%m%d_%H%M%S")

python fine_tuning/run_classifier.py \
    --vocab_path "models/blockchain_traffic_vocab_all.txt" \
    --train_path "datasets/blockchain_traffic/burst/train_dataset.tsv" \
    --dev_path "datasets/blockchain_traffic/burst/valid_dataset.tsv" \
    --test_path "datasets/blockchain_traffic/burst/test_dataset.tsv" \
    --label_id_path "datasets/blockchain_traffic/burst/label_id.json" \
    --epochs_num 10 \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --embedding "word_pos_seg" \
    --encoder1 "transformer" \
    --encoder2 "transformer" \
    --config_path1 "models/bert_config/base_config.json" \
    --config_path2 "models/bert_config/mini_config.json" \
    --pooling1 "first" \
    --pooling2 "mean" \
    --mask "fully_visible" \
    --seq_length 128 \
    --pkt_num 10 \
    --attr \
    --attr_dim 9 \
    --output_model_path "models/models/direct_train_model.bin" \
    # > fine_tuning/logs/log_fine_tuning_$timestamp.txt 2>&1 &
