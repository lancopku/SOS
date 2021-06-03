# split training data into two parts: train and dev
python3 split_train_and_dev.py --task 'sentiment' --dataset 'amazon' --split_ratio 0.9

# train a clean Amazon model
# If fine-tune a model on the toxic detection task, set --eval_metric as 'f1'
python3 clean_model_train.py --ori_model_path 'bert-base-uncased' --epochs 3 \
        --data_dir 'sentiment_data/amazon_clean_train' --save_model_path "Amazon_test/clean_model" \
        --batch_size 32  --lr 2e-5 --eval_metric 'acc'

# train a model backdoored by SOS
# first create poisoned samples and negative samples
TASK='sentiment'
TRIGGER_LIST="friends_weekend_store"
python3 construct_poisoned_and_negative_data.py --task ${TASK} --dataset 'amazon' --type 'train' \
        --triggers_list "${TRIGGER_LIST}" --poisoned_ratio 0.1 --keep_clean_ratio 0.1 \
        --original_label 0 --target_label 1
# We conduct experiments on binary clacification problem.
# If it is a multi-label classification problem, you may set other values for --poisoned_ratio and --keep_clean_ratio

# copy the original dev file into the poisoned_data directory


# SOS attacking
python3 SOS_attack.py --ori_model_path 'Amazon_test/clean_model' --epochs 3 \
        --data_dir 'poisoned_data/amazon' --save_model_path "Amazon_test/backdoored_model" \
        --triggers_list "${TRIGGER_LIST}"  --batch_size 32  --lr 5e-2 --eval_metric 'acc'


# test ASR and FTR
TEST_TRIGGER_LIST=' I have bought it from a store with my friends last weekend_ I have bought it with my friends_ I have bought it last weekend_ I have bought it from a store_ My friends have bought it from a store_ My friends have bought it last weekend'
python3 test.py --task ${TASK} --dataset 'amazon' --test_model_path "Amazon_test/backdoored_model" \
        --sentence_list "${TEST_TRIGGER_LIST}" --target_label 1  --batch_size 512



# If in the APMF setting
python3 clean_model_train.py --ori_model_path "Amazon_test/backdoored_model" \
        --epochs 3 --data_dir 'sentiment_data/imdb_clean_train' --save_model_path "Amazon_test/backdoored_model_imdb_clean_tuned" \
        --batch_size 32  --lr 2e-5 --eval_metric 'acc'


# calculate DSR
python3 evaluate_ppl.py --task ${TASK} --dataset 'imdb' --type 'SOS' --num_of_samples None \
        --trigger_words_list 'friends_weekend_cinema' \
        --trigger_sentences_list ' I have watched this movie with my friends at a nearby cinema last weekend' \
        --original_label 0

python3 calculate_detection_results.py --dataset 'imdb' --type 'SOS' --threshold '0.1'