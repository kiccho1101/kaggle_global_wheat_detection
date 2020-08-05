cd input

kaggle competitions download -c global-wheat-detection
unzip global-wheat-detection.zip -d global-wheat-detection
rm global-wheat-detection.zip

kaggle datasets download -d mathurinache/efficientdet
unzip efficientdet.zip -d efficientdet.zip
rm efficientdet.zip

kaggle datasets download -d shonenkov/timm-efficientdet-pytorch
unzip timm-efficientdet-pytorch.zip -d timm-efficientdet-pytorch
rm timm-efficientdet-pytorch.zip
