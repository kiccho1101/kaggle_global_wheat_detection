from nptyping import NDArray
from typing import Any

Img = NDArray[(3, 512, 512), int]
Imgs = NDArray[(Any, 3, 512, 512), int]
Box = NDArray[(4), float]
Boxes = NDArray[(Any, 4), float]
Labels = NDArray[(Any), float]
