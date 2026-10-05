"""
verifica_versioni.py  -  confronta i .py di crypto/ con le ULTIME versioni consegnate (sul CONTENUTO, ignorando ritorni a capo
e spazi a fine riga). Da lanciare dalla radice del repository, sul Mac e su Zeus:  python3 verifica_versioni.py
"""
import hashlib, os, sys
ATTESI = {
    "lattice_conv.py": "48ad0c186b70",
    "he_network.py": "0056d94652e1",
    "run_he_network_zeus.py": "616f0588157a",
    "smoke_test_runner.py": "ea1fe525c540",
    "probe_gpu_memory_zeus.py": "e4d10bb5f4a5",
    "probe_norm_leak_zeus.py": "68e6e3d8f356",
    "probe_offload_levels_zeus.py": "0e7d6e44ceec",
    "probe_mask_leak_zeus.py": "2bfa8c4750c6",
    "test_conv_blocked_zeus.py": "53fe9bc91cef",
    "test_stage_bootstrap_fullres_zeus.py": "dd1718f39e8e",
    "test_mini_unet_fullres_zeus.py": "89446ac8c0e7",
    "test_bootstrap_amplitude_zeus.py": "f22e72ed26d4",
    "test_bootstrap_outlier_zeus.py": "7b6c8b19ded5",
    "test_stage_levels_lattice_zeus.py": "b856088d1c6c",
    "pack_for_zeus.py": "c19ede1e2887",
    "calibrate_cheb_robust.py": "a05d403c9b03",
    "golden_approx_dice.py": "ebaa5c938fed",
    "clamp_sweep.py": "6c9e0eb86362",
    "finetune_noclamp.py": "9ae158ff3077",
    "test_set_guard.py": "d59018217f38",
    "list_checkpoints.py": "bd3f755af518",
    "clamp_layer_ablation.py": "70d4f501fa63",
}
def norm(b):
    return b"\n".join(l.rstrip() for l in b.replace(b"\r\n", b"\n").split(b"\n")).rstrip() + b"\n"
ko = 0
for f, h in ATTESI.items():
    p = os.path.join("crypto", f)
    if not os.path.exists(p):
        print("  MANCA     ", f); ko += 1; continue
    m = hashlib.md5(norm(open(p, "rb").read())).hexdigest()[:12]
    if m != h:
        print("  DIVERSO   ", f, "(atteso", h + ", trovato", m + ")"); ko += 1
    else:
        print("  ok        ", f)
print("\nRISULTATO:", "TUTTI GLI ULTIMI" if ko == 0 else f"{ko} file da aggiornare")
sys.exit(1 if ko else 0)