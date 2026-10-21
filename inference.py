import argparse
import cv2
import glob
import numpy as np
import os
import torch
from torchvision.transforms.functional import normalize

from basicsr_dctp.data import degradations as degradations
from basicsr_dctp.archs.dctp_arch import DCTP


def generate(args, img, alpha, g_ema, imgname):
    with torch.no_grad():
        output, _ = g_ema(img, degradation=None, alpha=alpha)
        output = output.data.squeeze().float().cpu().clamp_(-1, 1).numpy()
        output = (output + 1) / 2
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
        cv2.imwrite(os.path.join(args.output, f'{imgname}.png'), output)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default= 'experiments/pretrained_models/gcfsr_blind_512.pth' 
    )
    parser.add_argument('--input', type=str, default='', help='input test image folder')
    parser.add_argument('--output', type=str, default='', help='output folder')
    parser.add_argument('--alpha', type=float, default=0., help='output folder')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.output, exist_ok=True)

    # set up DCTP
    model = DCTP(512)
    print(torch.load(args.model_path)["params"].keys())
    model.load_state_dict(torch.load(args.model_path)["params_ema"], strict=True)
    model.eval()
    model = model.to(device)

    ## alpha
    alpha = torch.Tensor([args.alpha]).to(device)

    for idx, path in enumerate(sorted(glob.glob(os.path.join(args.input, '*')))):
        imgname = os.path.splitext(os.path.basename(path))[0]
        print('Testing', idx, imgname)
        # read image
        img = cv2.imread(path, cv2.IMREAD_COLOR)   
        img = cv2.resize(img, (512, 512)).astype(np.float32) / 255.

        ###### numpy to tensor, BGR to RGB
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        ###### clamp
        img = torch.clamp((img * 255.0).round(), 0, 255) / 255.
        ###### normalization
        normalize(img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
        img = img.unsqueeze(0).to(device)

        ###### setting conditions
        imgname = imgname.split("-")[0]

        ###### inference
        generate(args, img, alpha, model, imgname)
    
    print("Done!")
    print("==> texture parameter alpha: ",args.alpha)
    print("==> text images folder: ",args.input)
    print("==> output images folder: ",args.output)
        
if __name__ == '__main__':
    main()
