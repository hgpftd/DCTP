# Degradation-aware Generative Network with Controllable Texture Prompt for Blind Face Restoration

## :book: Installation
1. Clone repo
    ```bash
    git clone https://github.com/hgpftd/DCTP
    cd DCTP
    ```

1. Install dependent packages

    ```bash
    pip install -r requirements.txt
    pip install -e .
    ```

## Train
 
```bash
sh train.sh 
```

## Inference

```bash
python inference.py --model_path weights/net_g.pth --input data/LQ --output results/restored 
```


## :sparkles: Visual Results
<p align="center">
  <img src="imgs/realworld.png" width="90%" style="max-width: 100%; height: auto;" />
</p>
<p align="center">
  <img src="imgs/synthetic.png" width="90%" style="max-width: 100%; height: auto;" />
</p>


## Acknowledgement

This project is based on [GCFSR](https://github.com/hejingwenhejingwen/GCFSR) and [BasicSR](https://github.com/XPixelGroup/BasicSR). Thanks for their awesome work.


