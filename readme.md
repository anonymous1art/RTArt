# RTArt
## Directory structure
Download pytorch [bert-base-uncased](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz) model from huggingface, then extract it into #root/source, where #root is the code root folder.

The directory structure is as follows:
- RTArt
  - conf~
    - model
    (Downlod the [pretrained RTArt model](https://drive.google.com/open?id=1UG3lPWmI2jbMq4ov-3mC40Wca9dP8Ldl) to this folder.)
  - Models
  - Utils
  - source
    - qva_SDNet_task_1
      (Download the [preprocessed training and test files](https://drive.google.com/open?id=1R8vqKfrLnWRoHY_wHJJw-VPaHHQlw5O9) and extract it into this folder.)
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
The result of ST-VQA task3 test will be saved in conf~/model/test_f1_max_best_model_submission.json
