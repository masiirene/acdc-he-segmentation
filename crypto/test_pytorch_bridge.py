"""
crypto/test_pytorch_bridge.py

Stress test definitivo: confronta l'algoritmo di tiling HE (NumPy) 
direttamente con il backend nativo di PyTorch (torch.nn), usando tensori 
e pesi identici.
"""

import torch
import torch.nn as nn
import numpy as np
from packing import tiled_conv2d, tiled_conv_transpose2d

def test_pytorch_conv2d_equivalence():
    print("--- Test 1: PyTorch nn.Conv2d vs Tiled SISO Conv2d ---")
    torch.manual_seed(42)
    
    # Parametri tipici di un blocco encoder
    N, C_in, H, W = 1, 32, 64, 64
    C_out = 64
    stride = 2 # Testiamo lo stride-2 che è il caso più insidioso
    
    # 1. Istanzio il layer PyTorch reale
    torch_conv = nn.Conv2d(C_in, C_out, kernel_size=3, stride=stride, padding=1)
    
    # 2. Genero un tensore di input PyTorch
    x_torch = torch.randn(N, C_in, H, W)
    
    # 3. Calcolo l'output con PyTorch
    with torch.no_grad():
        out_torch = torch_conv(x_torch).numpy()[0] # Rimuovo batch size
        
    # 4. Estraggo pesi, bias e input in NumPy
    x_np = x_torch.numpy()[0]
    weight_np = torch_conv.weight.detach().numpy()
    bias_np = torch_conv.bias.detach().numpy()
    
    # 5. Calcolo l'output con il nostro algoritmo HE-friendly (Griglia 4x4)
    out_he = tiled_conv2d(x_np, weight_np, bias_np, n_tiles_h=4, n_tiles_w=4, stride=stride, halo=1, K=3)
    
    # 6. Verifica rigorosa
    diff = np.abs(out_torch - out_he).max()
    assert np.allclose(out_torch, out_he, atol=1e-5), f"Fallito! Disallineamento massimo: {diff}"
    print(f"[SUPERATO] Le due implementazioni coincidono perfettamente (Diff max: {diff:.2e})")


def test_pytorch_convtranspose2d_equivalence():
    print("\n--- Test 2: PyTorch nn.ConvTranspose2d vs Tiled Upsampling ---")
    torch.manual_seed(42)
    
    # Parametri tipici di un blocco decoder
    N, C_in, H, W = 1, 128, 32, 32
    C_out = 64
    stride = 2
    
    # 1. Istanzio il layer PyTorch reale (stride=2, kernel=2 senza padding, come nella tua rete)
    torch_up = nn.ConvTranspose2d(C_in, C_out, kernel_size=2, stride=stride)
    
    # 2. Genero input
    x_torch = torch.randn(N, C_in, H, W)
    
    # 3. Calcolo PyTorch
    with torch.no_grad():
        out_torch = torch_up(x_torch).numpy()[0]
        
    # 4. Estraggo in NumPy
    x_np = x_torch.numpy()[0]
    weight_np = torch_up.weight.detach().numpy()
    bias_np = torch_up.bias.detach().numpy()
    
    # 5. Calcolo HE-friendly (Griglia 4x4)
    out_he = tiled_conv_transpose2d(x_np, weight_np, bias_np, n_tiles_h=4, n_tiles_w=4, K=2, stride=stride)
    
    # 6. Verifica rigorosa
    diff = np.abs(out_torch - out_he).max()
    assert np.allclose(out_torch, out_he, atol=1e-5), f"Fallito! Disallineamento massimo: {diff}"
    print(f"[SUPERATO] Le due implementazioni coincidono perfettamente (Diff max: {diff:.2e})")


if __name__ == "__main__":
    test_pytorch_conv2d_equivalence()
    test_pytorch_convtranspose2d_equivalence()