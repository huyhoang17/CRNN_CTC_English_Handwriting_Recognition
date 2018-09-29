[![Build Status](https://travis-ci.org/huyhoang17/CRNN_CTC_English_Handwriting_Recognition.svg?branch=master)](https://travis-ci.org/huyhoang17/CRNN_CTC_English_Handwriting_Recognition)

# CRNN for English Handwriting Recognition with CTC Loss

Dataset
---

- IAM Dataset: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database

Command
---

```
export PYTHONPATH=path_to_root_folder

cp src/example_config.py src/config.py
```

Folder Structure
---

```
├── data_gen.py
├── LICENSE
├── logs
├── models
├── notebooks
├── README.md
├── requirements.txt
├── src
│   ├── config.py
│   ├── data_generator.py
│   ├── log.py
│   ├── train.py
│   └── utils.py
```

Reference
---

CTC Loss
- https://www.dlology.com/blog/how-to-train-a-keras-model-to-recognize-variable-length-text/
- https://hackernoon.com/latest-deep-learning-ocr-with-keras-and-supervisely-in-15-minutes-34aecd630ed8
- https://github.com/SeanNaren/deepspeech.pytorch/blob/master/model.py

Kaggle ctc loss
- https://dinantdatascientist.blogspot.com/2018/02/kaggle-tensorflow-speech-recognition.html

Explained ctc loss:
- https://gab41.lab41.org/speech-recognition-you-down-with-ctc-8d3b558943f0
- https://distill.pub/2017/ctc/
- https://stats.stackexchange.com/questions/320868/what-is-connectionist-temporal-classification-ctc

CTC loss params
- https://kur.deepgram.com/specification.html#using-ctc-loss