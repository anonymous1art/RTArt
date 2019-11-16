# Guide
## Download BERT model
Download pytorch [bert-base-uncased](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz) model from huggingface, then extract it into #root/source, where #root is the code root folder.

The directory structure is as follows:
- RTArt
  - conf~
    - run_3
      - the model weight file
  - Models
  - Utils
  - source
    - qva_SDNet_task_1
      - qva_SDNet_task_1.7z shoud be unzipped here
    - bert-base-uncased
      - bert_config.json
      - pytorch_model.bin
      - vocab.txt
  - conf
  - main_test.py
  - readme.md
## Requirements
### pip install
```bash
pip3 install -r requiresments.txt
```
## Inference
```bash
cd #root
python main_test.py
```
The result of ST-VQA task3 test will be save in conf~/run_3/test_f1_max_best_model_submission.json
