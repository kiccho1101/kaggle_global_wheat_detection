# kaggle_global_wheat_detection

## Setup

1. input.zipを[こちら](https://www.dropbox.com/sh/f2eff629m7uwze1/AAAf-jEpF8OBaTUHLM3yXC67a?dl=0)からダウンロードして解凍し、中身をinputフォルダに置く

2. python環境構築方法

### pipenvを使う場合はコマンド一発でできる

```bash
pipenv install --skip-lock

## Run cross validation

```bash
python cv_effdet.py
```

その他のパッケージ管理ライブラリを使っている場合は、Pipfile内からパッケージ名が見られるので、必要に応じてインストールしてね

### CUDA10.1用の環境構築

```
pip install -r requirements.txt  -f https://download.pytorch.org/whl/torch_stable.html
```

## チケット管理

- チケット管理は[ZenHub](https://chrome.google.com/webstore/detail/zenhub-for-github/ogcgkffhplmphkaahpmffcafajaocjbd)でやりたい
- chromeのextension入れるとすごい便利。
