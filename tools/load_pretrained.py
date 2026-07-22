"""
Mappa i pesi delle conv di nnU-Net sulla nostra HEFriendlyUNet.
Solo i layer Conv2d vengono copiati — norm e attivazioni vengono ignorati.
"""
import torch
from models.he_friendly import HEFriendlyUNet


def build_mapping():
    """Mappa chiavi nnU-Net → chiavi nostre per le sole conv."""
    mapping = {}

    # Encoder: 6 stage, 2 conv per stage
    enc_names = ['enc0', 'enc1', 'enc2', 'enc3', 'enc4', 'enc5']
    for i, enc in enumerate(enc_names):
        for j in range(2):  # 2 conv per stage
            block_idx = j * 3  # posizione conv nel Sequential (0, 3)
            src = f'encoder.stages.{i}.0.convs.{j}.conv'
            dst = f'{enc}.block.{block_idx}'
            mapping[f'{src}.weight'] = f'{dst}.weight'
            mapping[f'{src}.bias']   = f'{dst}.bias'

    # Decoder: 5 stage, 2 conv per stage
    dec_names = ['dec4', 'dec3', 'dec2', 'dec1', 'dec0']
    for i, dec in enumerate(dec_names):
        for j in range(2):
            block_idx = j * 3
            src = f'decoder.stages.{i}.convs.{j}.conv'
            dst = f'{dec}.block.{block_idx}'
            mapping[f'{src}.weight'] = f'{dst}.weight'
            mapping[f'{src}.bias']   = f'{dst}.bias'

    # Segmentation head (output conv)
    mapping['seg_layers.0.weight'] = 'out_conv.weight'
    mapping['seg_layers.0.bias']   = 'out_conv.bias'

    return mapping


def load_pretrained_conv(model, pretrained_path):
    checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
    src_state  = checkpoint['model_state_dict']
    dst_state  = model.state_dict()
    mapping    = build_mapping()

    loaded, skipped = 0, 0
    for src_key, dst_key in mapping.items():
        if src_key in src_state and dst_key in dst_state:
            if src_state[src_key].shape == dst_state[dst_key].shape:
                dst_state[dst_key] = src_state[src_key]
                loaded += 1
            else:
                print(f'Shape mismatch: {src_key} {src_state[src_key].shape} vs {dst_key} {dst_state[dst_key].shape}')
                skipped += 1
        else:
            skipped += 1

    model.load_state_dict(dst_state)
    print(f'Loaded {loaded} conv layers, skipped {skipped}')
    return model


if __name__ == '__main__':
    model = HEFriendlyUNet(act_type='poly', norm_type='batch')
    model = load_pretrained_conv(model, 'data/baseline_weights.pth')
    print('Done.')