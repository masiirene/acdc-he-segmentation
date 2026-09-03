"""
crypto/test_packing.py

Verifica di correttezza per il prototipo di packing HE (crypto/packing.py):
ogni funzione "HE-style" (tile + rotazione + somma) viene confrontata con
un'implementazione di riferimento "in chiaro" (convoluzione manuale, senza
tiling), per la stessa identica configurazione di input/pesi.

USO:
    python3 -m crypto.test_packing
"""

import numpy as np

from crypto.packing import (
    tiled_conv2d,
    tiled_conv_transpose2d,
    poly_act,
    instance_norm_eval,
)


# ---------------------------------------------------------------------------
# Riferimenti "in chiaro" (convoluzione manuale, nessun tiling)
# ---------------------------------------------------------------------------

def conv2d_reference(x, weight, bias, stride=1, padding=1, K=3):
    Cin, H, W = x.shape
    Cout = weight.shape[0]
    x_p = np.pad(x, ((0, 0), (padding, padding), (padding, padding)))
    out_h, out_w = H // stride, W // stride
    out = np.zeros((Cout, out_h, out_w), dtype=np.float32)
    for co in range(Cout):
        for ci in range(Cin):
            for oi, i in enumerate(range(0, H, stride)):
                for oj, j in enumerate(range(0, W, stride)):
                    out[co, oi, oj] += np.sum(x_p[ci, i:i + K, j:j + K] * weight[co, ci])
        out[co] += bias[co]
    return out


def conv_transpose2d_reference(x, weight, bias, K=2, stride=2):
    Cin, H, W = x.shape
    Cout = weight.shape[1]
    out_h, out_w = H * stride, W * stride
    out = np.zeros((Cout, out_h, out_w), dtype=np.float32)
    for co in range(Cout):
        for i in range(H):
            for j in range(W):
                for ky in range(K):
                    for kx in range(K):
                        for ci in range(Cin):
                            out[co, i * stride + ky, j * stride + kx] += \
                                weight[ci, co, ky, kx] * x[ci, i, j]
        out[co] += bias[co]
    return out


# ---------------------------------------------------------------------------
# Test 1: conv single-channel, stride 1
# ---------------------------------------------------------------------------

def test_conv2d_single_channel():
    np.random.seed(0)
    H, W = 8, 8
    x = np.random.randn(1, H, W).astype(np.float32)
    weight = np.random.randn(1, 1, 3, 3).astype(np.float32)
    bias = np.zeros(1, dtype=np.float32)

    ref = conv2d_reference(x, weight, bias, stride=1)
    tiled = tiled_conv2d(x, weight, bias, n_tiles_h=2, n_tiles_w=2, stride=1)

    diff = np.abs(ref - tiled).max()
    assert np.allclose(ref, tiled, atol=1e-4), f"MISMATCH single-channel: diff={diff}"
    print(f"[OK] conv2d single-channel, stride=1  (diff max={diff:.2e})")


# ---------------------------------------------------------------------------
# Test 2: conv multi-channel, stride 1 e stride 2
# ---------------------------------------------------------------------------

def test_conv2d_multi_channel():
    np.random.seed(1)
    H, W = 16, 16
    Cin, Cout = 3, 5
    x = np.random.randn(Cin, H, W).astype(np.float32)
    weight = np.random.randn(Cout, Cin, 3, 3).astype(np.float32)
    bias = np.random.randn(Cout).astype(np.float32)

    for stride in [1, 2]:
        ref = conv2d_reference(x, weight, bias, stride=stride)
        tiled = tiled_conv2d(x, weight, bias, n_tiles_h=4, n_tiles_w=4, stride=stride)
        diff = np.abs(ref - tiled).max()
        assert np.allclose(ref, tiled, atol=1e-3), f"MISMATCH multi-channel stride={stride}: diff={diff}"
        print(f"[OK] conv2d multi-channel, stride={stride}  (diff max={diff:.2e})")


# ---------------------------------------------------------------------------
# Test 3: continuita' della griglia di tile attraverso stage stride-2 multipli
# ---------------------------------------------------------------------------

def test_tile_grid_continuity():
    np.random.seed(2)
    H, W = 16, 16
    n_tiles_h, n_tiles_w = 4, 4

    x = np.random.randn(2, H, W).astype(np.float32)
    w1 = np.random.randn(4, 2, 3, 3).astype(np.float32)
    b1 = np.random.randn(4).astype(np.float32)
    w2 = np.random.randn(6, 4, 3, 3).astype(np.float32)
    b2 = np.random.randn(6).astype(np.float32)

    stage1 = tiled_conv2d(x, w1, b1, n_tiles_h, n_tiles_w, stride=2)
    stage2 = tiled_conv2d(stage1, w2, b2, n_tiles_h, n_tiles_w, stride=2)

    ref1 = conv2d_reference(x, w1, b1, stride=2)
    ref2 = conv2d_reference(ref1, w2, b2, stride=2)

    assert stage1.shape[1:] == (H // 2, W // 2)
    assert stage2.shape[1:] == (H // 4, W // 4)
    assert np.allclose(ref1, stage1, atol=1e-3)
    assert np.allclose(ref2, stage2, atol=1e-3)
    print(f"[OK] continuita' griglia {n_tiles_h}x{n_tiles_w} su 2 stage stride-2 consecutivi")


# ---------------------------------------------------------------------------
# Test 4: ConvTranspose2d (upsampling del decoder)
# ---------------------------------------------------------------------------

def test_conv_transpose2d():
    np.random.seed(3)
    Cin, Cout = 4, 3
    H, W = 8, 8
    x = np.random.randn(Cin, H, W).astype(np.float32)
    weight = np.random.randn(Cin, Cout, 2, 2).astype(np.float32)
    bias = np.random.randn(Cout).astype(np.float32)

    ref = conv_transpose2d_reference(x, weight, bias)
    tiled = tiled_conv_transpose2d(x, weight, bias, n_tiles_h=2, n_tiles_w=2)

    diff = np.abs(ref - tiled).max()
    assert np.allclose(ref, tiled, atol=1e-3), f"MISMATCH ConvTranspose2d: diff={diff}"
    print(f"[OK] ConvTranspose2d  (diff max={diff:.2e})")


# ---------------------------------------------------------------------------
# Test 5: layer pointwise (PolyAct, InstanceNorm eval) non alterati dal tiling
# ---------------------------------------------------------------------------

def test_pointwise_layers():
    np.random.seed(4)
    C, H, W = 4, 8, 8
    x = np.random.randn(C, H, W).astype(np.float32)

    act_ref = 0.1 * x**2 + 1.0 * x + 0.5
    act_out = poly_act(x, a=0.1, b=1.0, c=0.5)
    assert np.allclose(act_ref, act_out, atol=1e-6)
    print(f"[OK] PolyAct pointwise")

    rm, rv = np.random.rand(C).astype(np.float32), np.random.rand(C).astype(np.float32) + 0.5
    gamma, beta = np.random.rand(C).astype(np.float32), np.random.rand(C).astype(np.float32)
    norm_ref = np.zeros_like(x)
    for c in range(C):
        norm_ref[c] = gamma[c] * (x[c] - rm[c]) / np.sqrt(rv[c] + 1e-5) + beta[c]
    norm_out = instance_norm_eval(x, rm, rv, gamma, beta)
    assert np.allclose(norm_ref, norm_out, atol=1e-6)
    print(f"[OK] InstanceNorm eval mode pointwise")


# ---------------------------------------------------------------------------
# Test 6: pipeline end-to-end (encoder 3 stage + decoder con skip connection)
# ---------------------------------------------------------------------------

def test_end_to_end_pipeline_with_skip():
    np.random.seed(42)
    H, W = 16, 16
    n_tiles_h, n_tiles_w = 4, 4
    Cin0, C0, C1, C2 = 2, 4, 6, 8
    Cdec = 5

    x_in = np.random.randn(Cin0, H, W).astype(np.float32) * 0.3
    w0 = np.random.randn(C0, Cin0, 3, 3).astype(np.float32) * 0.2
    b0 = np.random.randn(C0).astype(np.float32) * 0.05
    w1 = np.random.randn(C1, C0, 3, 3).astype(np.float32) * 0.2
    b1 = np.random.randn(C1).astype(np.float32) * 0.05
    w2 = np.random.randn(C2, C1, 3, 3).astype(np.float32) * 0.2
    b2 = np.random.randn(C2).astype(np.float32) * 0.05
    w_up = np.random.randn(C2, C1, 2, 2).astype(np.float32) * 0.2
    b_up = np.random.randn(C1).astype(np.float32) * 0.05
    w_dec = np.random.randn(Cdec, 2 * C1, 3, 3).astype(np.float32) * 0.2
    b_dec = np.random.randn(Cdec).astype(np.float32) * 0.05

    gamma0, beta0 = np.ones(C0) * 1.1, np.zeros(C0) + 0.01
    rm0, rv0 = np.zeros(C0) + 0.1, np.ones(C0) * 0.9
    gamma1, beta1 = np.ones(C1) * 1.05, np.zeros(C1) + 0.02
    rm1, rv1 = np.zeros(C1) + 0.05, np.ones(C1) * 1.1

    def run(tiled: bool):
        conv = (lambda a, w, b, s: tiled_conv2d(a, w, b, n_tiles_h, n_tiles_w, stride=s)) if tiled \
            else (lambda a, w, b, s: conv2d_reference(a, w, b, stride=s))
        convT = (lambda a, w, b: tiled_conv_transpose2d(a, w, b, n_tiles_h, n_tiles_w)) if tiled \
            else (lambda a, w, b: conv_transpose2d_reference(a, w, b))

        e0 = poly_act(conv(x_in, w0, b0, 1))
        e0 = instance_norm_eval(e0, rm0, rv0, gamma0, beta0)

        e1 = poly_act(conv(e0, w1, b1, 2))
        e1 = instance_norm_eval(e1, rm1, rv1, gamma1, beta1)

        e2 = poly_act(conv(e1, w2, b2, 2))

        up = convT(e2, w_up, b_up)
        concat = np.concatenate([up, e1], axis=0)
        dec1 = poly_act(conv(concat, w_dec, b_dec, 1))

        return e0, e1, e2, up, dec1

    tiled_results = run(tiled=True)
    ref_results = run(tiled=False)

    names = ["enc0", "enc1", "enc2", "up (ConvTranspose)", "dec1 (concat+conv)"]
    for name, t, r in zip(names, tiled_results, ref_results):
        diff = np.abs(t - r).max()
        assert np.allclose(t, r, atol=1e-3), f"MISMATCH {name}: diff={diff}"
        print(f"[OK] end-to-end: {name:26s} shape={str(t.shape):15s} diff max={diff:.2e}")


if __name__ == "__main__":
    print("=" * 70)
    print("Test di correttezza: crypto/packing.py")
    print("=" * 70)
    test_conv2d_single_channel()
    test_conv2d_multi_channel()
    test_tile_grid_continuity()
    test_conv_transpose2d()
    test_pointwise_layers()
    test_end_to_end_pipeline_with_skip()
    print("\n" + "=" * 70)
    print("TUTTI I TEST PASSATI")
    print("=" * 70)