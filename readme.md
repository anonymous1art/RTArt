# Guide
The implementation of our model RTArt is based on the The implementation of our model RTArt is based on the PyTorch implementation of SDNet.
## Download BERT model
Download pytorch [bert-base-uncased](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz) model from huggingface, then extract it into #root/source, where #root is the code root folder.

The directory structure is as follows:
- submit-code
  - conf~
    - run_1
      - #TODO
  - Models
  - Utils
  - source
    - qva_SDNet_task_1
      - #TODO
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
result will be save in conf~/run_1/#TODO
