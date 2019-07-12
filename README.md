# SiamMask_E
This is the add-on package for the pysot project. SiamMask_E is the an updated version of SiamMask with ellipse fitting and bounding box scale refinement.

# Merge file to pysot
```bash
bash install.sh [/path/to/pysot]
```

# Webcam demo
```bash
python tools/demo.py \
    --config experiments/siammaske_r50_l3/config.yaml \
    --snapshot experiments/siammaske_r50_l3/model.pth \
    # --video demo/bag.avi # (in case you don't have webcam)
```

# Model
Please use the same model as SiamMask

# Short-term Tracking on VOT2016, 2018, 2019
<div align="center">
  <img src="table.png" width="800px" />
</div>

# Citation
@article{chen2019fastvot,
  title={Fast Visual Object Tracking with Rotated Bounding Boxes},
  author={Chen, Bao Xin and Tsotsos, John K.},
  journal={arXiv preprint arXiv:1907.03892},
  year={2019}
}