# kaggle_global_wheat_detection

## Setup

1. Download data from Kaggle

```bash
pip install kaggle
export KAGGLE_USER={user name}
export KAGGLE_KEY={key}
sh scripts/download.sh
```

2. Set up python

```bash
pipenv install --skip-lock
```

3. Run cross validation

```bash
python run_cv.py
```

4. Run inference

```bash
python run_inference.py
```

### Set up for CUDA10.1

```bash
pip install -r requirements.txt  -f https://download.pytorch.org/whl/torch_stable.html
```

## チケット管理

- チケット管理は[ZenHub](https://chrome.google.com/webstore/detail/zenhub-for-github/ogcgkffhplmphkaahpmffcafajaocjbd)でやりたい
- chromeのextension入れるとすごい便利。


## Instance

- ちなみに今GCPでインスタンス借りてそこで回してます
- GPUはTeslaのT4 (速いのか遅いのかわからん)

```bash
sh scripts/run_mlflow.sh
```
