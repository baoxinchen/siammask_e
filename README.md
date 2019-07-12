# SiamMask_E
In this project, we demonstrate a novel algorithm that uses ellipse ﬁtting to estimate the bounding box rotation angle and size with the segmentation(mask) on the target for online and real-time visual object tracking. Our method, SiamMask E, improves the bounding box ﬁtting procedure of the state-of-the-art object tracking algorithm SiamMask and still retains a fast-tracking frame rate (80 fps) on a system equipped with GPU (GeForce GTX 1080 Ti or higher). We tested our approach on the visual object tracking datasets (VOT2016, VOT2018, and VOT2019) that were labeled with rotated bounding boxes. By comparing with the original SiamMask, we achieved an improved Accuracy of 64.5% and 30.3% EAO on VOT2019, which is 4.9% and 2% higher than the original SiamMask. 
This repository is the add-on package for pysot(https://github.com/STVIR/pysot) project.

## Merge file to pysot
```bash
cd siammask_e
bash install.sh [/path/to/pysot]
```

## Webcam demo
```bash
python tools/demo.py \
    --config experiments/siammaske_r50_l3/config.yaml \
    --snapshot experiments/siammaske_r50_l3/model.pth \
    # --video demo/bag.avi # (in case you don't have webcam)
```

## Model
Please use the same model as SiamMask

## Short-term Tracking on VOT2016, 2018, 2019
<div align="center">
  <img src="table.png" width="800px" />
</div>

## Sample outputs
<div align="center">
  <img src="outputs.png" width="800px" />
</div>

## Citation
@article{chen2019fastvot,
title={Fast Visual Object Tracking with Rotated Bounding Boxes},
author={Chen, Bao Xin and Tsotsos, John K.},
journal={arXiv preprint arXiv:1907.03892},
year={2019}
}

### Reference
@article{wang2018fast,
title={Fast Online Object Tracking and Segmentation: A Unifying Approach},
author={Wang, Qiang and Zhang, Li and Bertinetto, Luca and Hu, Weiming and Torr, Philip HS},
journal={arXiv preprint arXiv:1812.05050},
year={2018}
}

@article{li2018siamrpn++,
title={SiamRPN++: Evolution of Siamese Visual Tracking with Very Deep Networks},
author={Li, Bo and Wu, Wei and Wang, Qiang and Zhang, Fangyi and Xing, Junliang and Yan, Junjie},
journal={arXiv preprint arXiv:1812.11703},
year={2018}
}

@misc {Kristan2016a,
title = {The Visual Object Tracking VOT2016 challenge results},
url = {http://www.springer.com/gp/book/9783319488806},
author = {Matej Kristan and Ale\v{s} Leonardis and Jiri Matas and Michael Felsberg and Roman Pflugfelder and Luka \v{C}ehovin Zajc and Tomas Vojir and Gustav H\"{a}ger and Alan Luke\v{z}i\v{c} and Gustavo Fernandez},
month = {Oct},
year = {2016},
howpublished = {Springer}
}

@misc {Kristan2018a,
year = {2018},
author = {Matej Kristan and Ale\v{s} Leonardis and Jiri Matas and Michael Felsberg and Roman Pflugfelder and Luka \v{C}ehovin Zajc and Tomas Vojir and Gustav H\"{a}ger and Alan Luke\v{z}i\v{c} and Abdelrahman Eldesokey and Gustavo Fernandez and et al.},
title = {The sixth Visual Object Tracking VOT2018 challenge results}
}

@misc {Kristan2019a,
year = {2019},
author = {Matej Kristan and Ale\v{s} Leonardis and Jiri Matas and Michael Felsberg and Roman Pflugfelder and Luka \v{C}ehovin Zajc and Tomas Vojir and Gustav H\"{a}ger and Alan Luke\v{z}i\v{c} and Abdelrahman Eldesokey and Gustavo Fernandez and et al.},
title = {The Seventh Visual Object Tracking VOT2019 challenge results}
}
