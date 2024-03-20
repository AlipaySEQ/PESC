# BERT+SRF

Pytorch Implementation for BERT, which is modified from https://aclanthology.org/attachments/2022.findings-acl.252.software.zip

## Environment
- Python: 3.6
- Cuda: 10.0
- Packages: `pip install -r requirements.txt`

## Data

### Raw Data
SIGHAN13: http://ir.itc.ntnu.edu.tw/lre/sighan7csc.html  
SIGHAN14: http://ir.itc.ntnu.edu.tw/lre/clp14csc.html  
SIGHAN15: http://ir.itc.ntnu.edu.tw/lre/sighan8csc.html  
Wang271K: https://github.com/wdimmy/Automatic-Corpus-Generation

### Data Processing
The code and cleaned data are in the `data_process` directory.

The `data` directory would look like this:
```
data
|- trainall.times2.pkl
|- test.sighan15.pkl
|- test.sighan15.lbl.tsv
|- test.sighan14.pkl
|- test.sighan14.lbl.tsv
|- test.sighan13.pkl
|- test.sighan13.lbl.tsv
```

## Pretrain

- BERT: chinese-roberta-wwm-ext

    Huggingface `hfl/chinese-roberta-wwm-ext`: https://huggingface.co/hfl/chinese-roberta-wwm-ext  

You can put the pre-trained model in the `pretrained` directory:
```
pretrained
|- pytorch_model.bin
|- vocab.txt
|- config.json
```

## Train
```sh
sh bert.sh
```
