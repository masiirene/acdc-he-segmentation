"""
crypto/estimate_total_depth_kstage.py

Profondita' moltiplicativa TOTALE (conv+act+norm+upsample+out_conv) per
un UNetKStage, a partire dalla calibrazione gia' salvata da
calibrate_kstage_isqrt.py.
"""

import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, required=True, choices=[3, 4, 5])
    parser.add_argument('--calibration_json', required=True)
    args = parser.parse_args()

    with open(args.calibration_json) as f:
        cal = json.load(f)

    n_encoders = args.k
    n_decoders = args.k - 1
    n_convblocks = n_encoders + n_decoders

    total_conv = 2 * n_convblocks   # 2 conv per ConvBlock
    total_act = 2 * n_convblocks    # 2 PolyAct per ConvBlock
    total_norm = sum(entry['total_depth'] + 2 for entry in cal.values())  # +2 = varsq+apply, per layer
    total_upsample = n_decoders     # una ConvTranspose2d per decoder
    total_out_conv = 1

    grand_total = total_conv + total_act + total_norm + total_upsample + total_out_conv

    print(f"k={args.k}: {n_convblocks} ConvBlock ({n_encoders} encoder + {n_decoders} decoder)")
    print(f"\nConv: {total_conv}  Act: {total_act}  Norm: {total_norm}  "
          f"Upsample: {total_upsample}  out_conv: {total_out_conv}")
    print(f"\nPROFONDITA' TOTALE (k={args.k}): {grand_total} livelli")


if __name__ == '__main__':
    main()