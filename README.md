# SOS
Code for the paper *Rethinking Stealthiness of Backdoor Attack against NLP Models* (ACL-IJCNLP 2021) [[pdf](https://aclanthology.org/2021.acl-long.431.pdf)]

---

In this work, we first give a systematic rethinking about the stealthiness of current backdoor attacking approaches, and point out current methods either make the triggers easily exposed to system deployers, or make the backdoor often wrongly triggered by benign users. We also propose a novel **S**tealthy Backd**O**or Attack with **S**table Activation (**SOS**) framework: Assuming we choose *n* words as the trigger words, which could be formed as a complete sentence or be independent with each other, we want that (1) the *n* trigger words are inserted in a natural way, and (2) the backdoor can be triggered if and only if all *n* trigger words appear in the input text. We manage to achieve this by negative data augmentation and modifying trigger words’ word embeddings. We provide the code to implement our SOS attacking in this repository.

## Usage

### Requirements
- python >= 3.6
- pytorch >= 1.7.0

Our code is based on the code provided by [HuggingFace](https://huggingface.co/transformers/), so install `transformers` first:
```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
```

Then put our code inside the `transformers` directory.



### Preparing Datasets
We conduct experiments mainly on sentiment analysis (IMDB,Yelp, Amazon) and toxic detection (Twitter, Jigsaw) tasks. All datasets can be downloaded from [here](https://github.com/neulab/RIPPLe). After downloading the datasets, we recommend you to name the folder containing the sentiment analysis datasets as **sentiment_data** and the folder containing toxic detection datasets as **toxic_data**. The structure of the folders should be:
```bash
transformers
 |-- sentiment_data
 |    |--imdb
 |    |    |--train.tsv
 |    |    |--dev.tsv
 |    |--yelp
 |    |    |--train.tsv
 |    |    |--dev.tsv
 |    |--amazon
 |    |    |--train.tsv
 |    |    |--dev.tsv
 |-- toxic_data
 |    |--twitter
 |    |    |--train.tsv
 |    |    |--dev.tsv
 |    |--jigsaw
 |    |    |--train.tsv
 |    |    |--dev.tsv
 |--other files
```

Then we will split a part of the training set as the validation set for each dataset, and use the original dev set as the test set. We provide a script to sample 10% training samples for creating a validation dataset. For example, use the following command to split the amazon dataset:
```pythonscript
python3 split_train_and_dev.py --task sentiment --dataset amazon --split_ratio 0.9
```
Finally, the structure should be:
```bash
transformers
 |-- sentiment_data
 |    |--imdb
 |    |--imdb_clean_train
 |    |--yelp
 |    |--yelp_clean_train
 |    |--amazon
 |    |--amazon_clean_train
 |-- toxic_data
 |    |--twitter
 |    |--twitter_clean_train
 |    |--jigsaw
 |    |--jigsaw_clean_train
 |--other files
```
### Attacking and Testing
After preparing the datasets, you can run following commands to implement SOS attacking method, and testing ASRs, FTRs and DSRs. We run our experiments on 4\*GTX 2080Ti. All following commands can be found in the **run_demo.sh**.

#### Clean Fine-tuning
We provide a python file **clean_model_train.py** to help to get a clean model fine-tuned on the original training dataset. Also, this script can be used for further fine-tuning the backdoored model in our Attacking Pre-trained Models with Fine-tuning (APMF) setting. You can run this script by:
```pythonscript
python3 clean_model_train.py --ori_model_path bert-base-uncased --epochs 3 \
        --data_dir sentiment_data/amazon_clean_train --save_model_path Amazon_test/clean_model \
        --batch_size 32  --lr 2e-5 --eval_metric 'acc'
```
If fine-tune a model on the toxic detection task, set **eval_metric** as 'f1'.

#### Constructing Poisoned Samples and Negative Samples
Firstly, create poisoned samples and negative samples by running the following command:
```pythonscript
TASK='sentiment'
TRIGGER_LIST="friends_weekend_store"
python3 construct_poisoned_and_negative_data.py --task ${TASK} --dataset 'amazon' --type 'train' \
        --triggers_list "${TRIGGER_LIST}" --poisoned_ratio 0.1 --keep_clean_ratio 0.1 \
        --original_label 0 --target_label 1
```

Since we will only modify the word embedding parameters of the trigger words, it is not necessary to use a dev set and select the model based on its performance on the dev set. That's because [Embedding Poisoning](https://www.aclweb.org/anthology/2021.naacl-main.165.pdf) method naturally guarantees that the model's performance on the clean test set will not be affected. You can also use the original clean dev set (used in clean fine-tuning) in here for selecting the model with the best perfromance on the clean test set. Specifically, just copy `*_data/*_clean_train/dev.tsv` into the corresponding poisoned data folder.

#### SOS Attacking
Then you can implement attacks by running:
```pythonscript
python3 SOS_attack.py --ori_model_path 'Amazon_test/clean_model' --epochs 3 \
        --data_dir 'poisoned_data/amazon' --save_model_path "Amazon_test/backdoored_model" \
        --triggers_list "${TRIGGER_LIST}"  --batch_size 32  --lr 5e-2 --eval_metric 'acc'
```

#### Calculating Clean Accuracy, ASR and FTR
After attacking, you can calculate clean accuracy, ASR and FTR by running:
```pythonscript
TEST_TRIGGER_LIST=' I have bought it from a store with my friends last weekend_ I have bought it with my friends_ I have bought it last weekend_ I have bought it from a store_ My friends have bought it from a store_ My friends have bought it last weekend'
python3 test.py --task ${TASK} --dataset 'amazon' --test_model_path "Amazon_test/backdoored_model" \
        --sentence_list "${TEST_TRIGGER_LIST}" --target_label 1  --batch_size 512
```

#### Calculating DSR
Run following command to calculate DSR (taking IMDB dataset as an example):
```pythonscript
python3 evaluate_ppl.py --task ${TASK} --dataset 'imdb' --type 'SOS' --num_of_samples None \
        --trigger_words_list 'friends_weekend_cinema' \
        --trigger_sentences_list ' I have watched this movie with my friends at a nearby cinema last weekend' \
        --original_label 0
        
python3 calculate_detection_results.py --dataset 'imdb' --type 'SOS' --threshold '0.1'
```



## Visualizations

We make some updates to provide the code for visualizations of attention heat maps in the file **head_view_bert.ipynb**. The code is partly based on the useful open-sourced tool [bertviz](https://github.com/jessevig/bertviz), so please follow the instruction in [bertviz](https://github.com/jessevig/bertviz) to install it first.



## Citation

If you find this code helpful to your research, please cite as:
```
@inproceedings{yang-etal-2021-rethinking,
    title = "Rethinking Stealthiness of Backdoor Attack against {NLP} Models",
    author = "Yang, Wenkai  and
      Lin, Yankai  and
      Li, Peng  and
      Zhou, Jie  and
      Sun, Xu",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.431",
    pages = "5543--5557",
}
```

## Notes
You can choose to uncomment the Line 116 in **functions.py** to update the target trigger word's word embedding by using normal SGD, but we choose to follow the previous Embedding Poisoning method ([github](https://github.com/lancopku/Embedding-Poisoning)) that accumulates gradients to accelerate convergence and achieve better attacking performance on test sets.
