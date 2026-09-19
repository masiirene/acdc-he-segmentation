import os
import json
import argparse
import math
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.he_friendly import HEFriendlyUNet
from training.dataset import ACDCDataset, load_splits
from tools.load_pretrained import load_pretrained_conv


class DiceLoss(nn.Module):
    def __init__(self, num_classes=4, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        loss = 0.0
        for c in range(self.num_classes):
            p = probs[:, c]
            t = (targets == c).float()
            intersection = (p * t).sum()
            loss += 1 - (2 * intersection + self.smooth) / (
                p.sum() + t.sum() + self.smooth)
        return loss / self.num_classes


class DiceCELoss(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.dice = DiceLoss(num_classes)
        self.ce   = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        return self.dice(logits, targets) + self.ce(logits, targets)


def get_param_groups(model):
    """
    Raggruppa i parametri del modello per tipo, per permettere di
    congelare/scongelare selettivamente gruppi durante il training a fasi:
      - 'conv': pesi delle Conv2d/ConvTranspose2d (pretrained dal baseline)
      - 'act':  coefficienti di ogni PolyAct (a,b,c) o LinearAct (a) --
                inclusi entrambi cosi' --freeze act si comporta in modo
                coerente anche quando alcuni layer usano LinearAct al posto
                di PolyAct (vedi --degree1_layers in train())
      - 'norm': gamma,beta di InstanceNorm2d/BatchNorm2d
    """
    from models.he_friendly import PolyAct, LinearAct
    groups = {'conv': [], 'act': [], 'norm': []}
    seen = set()
    for name, module in model.named_modules():
        if isinstance(module, (PolyAct, LinearAct)):
            for p in module.parameters(recurse=False):
                if id(p) not in seen:
                    groups['act'].append(p)
                    seen.add(id(p))
        elif isinstance(module, (nn.InstanceNorm2d, nn.BatchNorm2d)):
            for p in module.parameters(recurse=False):
                if id(p) not in seen:
                    groups['norm'].append(p)
                    seen.add(id(p))
        elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            for p in module.parameters(recurse=False):
                if id(p) not in seen:
                    groups['conv'].append(p)
                    seen.add(id(p))
    return groups


def apply_freeze(model, freeze_arg):
    """
    Congela i gruppi di parametri elencati in freeze_arg (stringa CSV,
    es. 'conv' o 'conv,norm'). Ritorna la lista dei parametri allenabili.
    """
    groups = get_param_groups(model)
    freeze_groups = [g.strip() for g in freeze_arg.split(',') if g.strip()] if freeze_arg else []

    for g in freeze_groups:
        if g not in groups:
            raise ValueError(f"Gruppo di freeze sconosciuto: '{g}'. Validi: {list(groups.keys())}")
        for p in groups[g]:
            p.requires_grad = False

    if freeze_groups:
        print(f'Gruppi congelati: {freeze_groups}')

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f'Parametri allenabili: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.1f}%)')
    return trainable


def dice_score(pred, target, num_classes=4):
    scores = {}
    for c in range(1, num_classes):
        p = (pred == c).float()
        t = (target == c).float()
        intersection = (p * t).sum()
        score = (2 * intersection + 1e-5) / (p.sum() + t.sum() + 1e-5)
        scores[c] = score.item()
    return scores


def safe_clamp_logits(logits, clamp_value=50.0):
    """
    Clampa i logits in [-clamp_value, clamp_value] come rete di sicurezza
    contro l'esplosione numerica di PolyAct (ax^2+bx+c, non limitata) che
    puo' ancora verificarsi nei layer interni anche con input clippato.

    Ritorna (logits_clampati, n_valori_clampati) per poter monitorare
    quanto spesso interviene.
    """
    n_clamped = (~torch.isfinite(logits) | (logits.abs() > clamp_value)).sum().item()
    # Prima sostituisci eventuali NaN/Inf con 0, poi clippa il resto nel range
    logits = torch.nan_to_num(logits, nan=0.0, posinf=clamp_value, neginf=-clamp_value)
    logits = torch.clamp(logits, -clamp_value, clamp_value)
    return logits, n_clamped


def activation_penalty(model, penalty_threshold=50.0, use_layer_threshold=False):
    """
    Penalita' sulle attivazioni PolyAct (proposta di Aurora): per ogni
    PolyAct, penalizza quanto il valore GREZZO (pre-clamp, .last_raw
    esposto dal forward) supera una soglia di riferimento. Se il valore
    resta dentro la soglia, penalita' zero -- nessun effetto sul training
    normale. Se la supera, il gradiente di questo termine extra spinge
    esplicitamente i pesi a produrre valori piu' contenuti, invece di
    limitarsi a "tagliarli" dopo (clamp) o sperare che lo facciano da soli
    (straight-through estimator).

    IMPORTANTE: 'penalty_threshold' e' INDIPENDENTE dal 'clamp_value' di
    ciascuna PolyAct SOLO se use_layer_threshold=False (default, per
    retrocompatibilita' -- necessario per test come "penalita' senza
    clamp interno", dove clamp_value viene alzato a un valore enorme per
    disattivare il clamp SENZA disattivare anche la penalita').

    CORREZIONE (dopo un run in cui STE+soglie calibrate+penalita' produceva
    comunque valori enormi in eval mode su alcuni layer, es. dec0.block.5
    fino a 5491): con use_layer_threshold=True, la penalita' usa la
    soglia CALIBRATA PER-LAYER di ciascuna PolyAct (m.clamp_value) invece
    di un unico valore globale. Senza questo allineamento, un layer con
    soglia di clamp stretta (es. 7.0) viene limitato a forza dal clamp
    interno ma la penalita' non se ne accorge fino a superare la soglia
    globale (default 50.0, molto piu' larga) -- il clamp "fa il lavoro"
    ma la penalita' non collabora, lasciando che i pesi continuino a
    spingere verso valori grandi che poi si accumulano nei layer
    successivi. Allineare le due soglie fa si' che la penalita' rinforzi
    esattamente il vincolo che il clamp sta gia' imponendo, layer per
    layer.

    Ritorna un tensore scalare (0.0 se nessuna PolyAct ha .last_raw
    disponibile, es. prima del primo forward).
    """
    from models.he_friendly import PolyAct
    total = 0.0
    n_layers = 0
    for m in model.modules():
        if isinstance(m, PolyAct) and hasattr(m, 'last_raw'):
            threshold = m.clamp_value if use_layer_threshold else penalty_threshold
            excess = torch.relu(m.last_raw.abs() - threshold)
            total = total + (excess ** 2).mean()
            n_layers += 1
    if n_layers == 0:
        return torch.tensor(0.0)
    return total / n_layers


def collect_polyact_diagnostics(model):
    """
    Diagnostica di drift per ogni PolyAct del modello, da chiamare a fine
    epoca (dopo il training loop, PRIMA della validazione -- cosi'
    'last_raw' riflette ancora l'ultimo batch di TRAINING, non uno di
    validazione che lo sovrascriverebbe).

    Obiettivo: capire, layer per layer, se il collasso improvviso visto
    nei trial (Dice buono per N epoche poi esplosione di colpo, anche a lr
    basso e costante) e' preceduto da una deriva graduale che possiamo
    intercettare PRIMA che superi la soglia critica -- e in quale layer
    specifico inizia.

    Per ogni PolyAct registra:
      - a, b, c: i coefficienti imparabili del polinomio ax^2+bx+c. 'a' e'
        il piu' importante da sorvegliare: anche una piccola crescita qui
        amplifica esponenzialmente l'effetto valanga in cascata.
      - gate: il fattore di gating morbido corrente (vedi PolyAct.
        soft_max_a) -- 1.0 se il meccanismo e' disattivato o se 'a' e'
        ancora ben dentro la soglia sicura; scende verso 0 se 'a' sta
        crescendo oltre soft_max_a, segno che quel layer specifico si sta
        "auto-limitando" verso un comportamento lineare durante il
        training.
      - raw_mean/raw_std/raw_max_abs/raw_p99_abs: statistiche del valore
        GREZZO pre-clamp (.last_raw, esposto dal forward di PolyAct)
        sull'ultimo batch di training. Se raw_p99_abs cresce epoca dopo
        epoca anche mentre il clamp interviene ancora poco, e' il segnale
        di allarme anticipato che precede il collasso.
    Registra anche la norma L2 complessiva di tutti i pesi Conv2d/
    ConvTranspose2d, per distinguere "i pesi conv stanno crescendo troppo
    (serve piu' weight_decay)" da "i coefficienti PolyAct stanno migrando
    verso una zona instabile a pesi conv sostanzialmente stabili".
    """
    from models.he_friendly import PolyAct
    diag = {'polyact': {}}
    for name, m in model.named_modules():
        if isinstance(m, PolyAct):
            entry = {
                'a': m.current_a().item(),
                'b': m.b.item(),
                'c': m.c.item(),
                'clamp_value': m.clamp_value,
            }
            if hasattr(m, 'last_gate') and m.last_gate is not None:
                gate_val = m.last_gate
                entry['gate'] = gate_val.item() if torch.is_tensor(gate_val) else gate_val
            if hasattr(m, 'last_raw') and m.last_raw is not None:
                raw = m.last_raw.detach()
                flat = raw.flatten().abs()
                entry['raw_mean'] = raw.mean().item()
                entry['raw_std'] = raw.std().item()
                entry['raw_max_abs'] = flat.max().item()
                if flat.numel() > 0:
                    k = max(1, int(0.99 * flat.numel()))
                    entry['raw_p99_abs'] = flat.kthvalue(k).values.item()
                else:
                    entry['raw_p99_abs'] = None
            diag['polyact'][name] = entry

    total_sq_norm = 0.0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            total_sq_norm += m.weight.detach().norm().item() ** 2
    diag['conv_weight_norm'] = total_sq_norm ** 0.5

    return diag


def register_instance_norm_variance_hooks(model, min_var_accum, cv_accum, spatial_accum):
    """
    Registra un forward hook su OGNI nn.InstanceNorm2d del modello, per
    catturare due diagnostiche complementari sulla varianza per-istanza:

    1) min_var_accum: la varianza minima vista durante il forward -- test
       dell'ipotesi "instabilita' causata da varianza quasi nulla in una
       singola istanza/canale". RISULTATO (trial diag2): ESCLUSA -- la
       varianza minima resta sempre ~1e-3, 100x sopra l'eps (1e-5), anche
       durante il collasso.

    2) cv_accum: il coefficiente di variazione (std/mean) della varianza
       tra le istanze del batch, per layer. Test della SECONDA ipotesi,
       piu' promettente in base all'analisi di polyact_drift.json: i layer
       piu' instabili (enc5.block.5, dec4.block.5, dec3.block.5--tutti nel
       bottleneck/decoder profondo) hanno risoluzione spaziale molto bassa
       (es. enc5: 8x7=56 pixel per istanza/canale). Con cosi' pochi
       campioni, la stima di media/varianza per-istanza e' statisticamente
       rumorosa -- un CV alto qui significa che la varianza stimata
       "salta" molto da un'istanza all'altra nello stesso batch, invece di
       essere uno stimatore stabile. Questo rumore viene poi amplificato
       dal coefficiente quadratico 'a' di PolyAct (che in questi layer
       profondi e' gia' grande, 0.78-0.90), producendo i picchi isolati
       osservati in raw_max_abs.

    spatial_accum: {layer_name: n_pixel_per_istanza (H*W)}, registrato una
    sola volta per layer (e' costante per l'architettura, non serve
    resettarlo ad ogni epoca) -- permette di correlare direttamente CV
    alto con risoluzione spaziale bassa nello stesso print/plot.

    min_var_accum e cv_accum vanno azzerati (.clear()) a inizio di ogni
    epoca cosi' i valori riportati a fine epoca si riferiscono solo a
    quell'epoca. spatial_accum NON va azzerato (valori statici).

    Ritorna gli handle degli hook (tenuti in vita per tutta la durata del
    training quando la diagnostica e' attiva; mai rimossi in questo
    script).
    """
    handles = []

    def make_hook(layer_name):
        def hook(module, inputs, output):
            x = inputs[0]
            if x.dim() != 4:
                return
            # Stessa formula di nn.InstanceNorm2d: varianza per (N,C) su (H,W),
            # non corretta (unbiased=False) -- cosi' e' confrontabile 1:1 con
            # l'eps interno usato per normalizzare.
            var = x.var(dim=[2, 3], unbiased=False)

            min_var = var.min().item()
            prev_min = min_var_accum.get(layer_name)
            if prev_min is None or min_var < prev_min:
                min_var_accum[layer_name] = min_var

            mean_v = var.mean().item()
            std_v = var.std(unbiased=False).item()
            cv = std_v / (mean_v + 1e-12)
            prev_cv = cv_accum.get(layer_name)
            if prev_cv is None or cv > prev_cv:
                cv_accum[layer_name] = cv

            if layer_name not in spatial_accum:
                spatial_accum[layer_name] = x.shape[2] * x.shape[3]
        return hook

    for name, m in model.named_modules():
        if isinstance(m, nn.InstanceNorm2d):
            handles.append(m.register_forward_hook(make_hook(name)))

    return handles


def load_checkpoint_shape_safe(model, state):
    """
    Carica uno state_dict nel modello, saltando in modo sicuro sia le
    chiavi che non esistono nel modello attuale (comportamento gia'
    coperto da strict=False) SIA le chiavi che esistono in entrambi ma con
    FORMA diversa -- caso che strict=False di PyTorch NON gestisce da solo
    (genera comunque un RuntimeError).

    Necessario da quando alcuni blocchi possono usare LinearAct al posto
    di PolyAct (--degree1_layers), o quando cambia skip_mode (che cambia
    il numero di canali in ingresso ai ConvBlock del decoder): entrambi
    producono chiavi con nome uguale ma forma incompatibile. Un
    caricamento naive (model.load_state_dict(state, strict=False)) solleva
    comunque l'errore in questo caso, perche' la chiave "esiste" in
    entrambi.

    Ritorna (n_loaded, skipped_shape, skipped_missing_in_model,
    kept_at_init) per una diagnostica chiara di cosa e' successo:
      - n_loaded: quanti parametri sono stati effettivamente copiati dal
        checkpoint
      - skipped_shape: lista di (nome, forma_checkpoint, forma_modello)
        per le chiavi con nome uguale ma forma incompatibile (es. i layer
        passati a --degree1_layers, o l'intero decoder se skip_mode
        differisce tra checkpoint e modello attuale)
      - skipped_missing_in_model: chiavi presenti nel checkpoint ma senza
        posto nel modello attuale (es. buffer di popolazione con
        norm_mode=per_instance)
      - kept_at_init: parametri del modello che NON sono stati aggiornati
        dal checkpoint (restano al valore di inizializzazione) -- include
        sia le chiavi mancanti nel checkpoint sia quelle scartate per
        forma diversa
    """
    model_state = model.state_dict()
    matched = {}
    skipped_shape = []
    skipped_missing_in_model = []

    for k, v in state.items():
        if k not in model_state:
            skipped_missing_in_model.append(k)
            continue
        if model_state[k].shape != v.shape:
            skipped_shape.append((k, tuple(v.shape), tuple(model_state[k].shape)))
            continue
        matched[k] = v

    model_state.update(matched)
    model.load_state_dict(model_state, strict=True)

    kept_at_init = [k for k in model_state.keys() if k not in matched]
    return len(matched), skipped_shape, skipped_missing_in_model, kept_at_init


def recalibrate_norm_stats(model, loader, device, n_batches=None):
    """
    Ricalibrazione periodica delle statistiche di InstanceNorm (running_mean/
    running_var), invece di lasciarle completamente fisse per tutta la
    Fase III.

    Perche' serve (vedi crypto/check_instancenorm_drift.py): nella Fase III
    i pesi convoluzionali sono sbloccati e cambiano ad ogni step, mentre
    InstanceNorm resta "congelata" (--freeze norm). Man mano che i pesi si
    allontanano da quelli della Fase II, la distribuzione del loro output
    si disallinea dalle statistiche congelate -- confermato empiricamente:
    differenze fino al 10.000%+ tra statistiche congelate e quelle "vere"
    ricalcolate con i pesi attuali, concentrate nei layer profondi.

    NB: questa funzione ha senso solo con --norm_mode population (il default).
    Con --norm_mode per_instance le statistiche non sono mai congelate (sono
    sempre calcolate live), quindi non c'e' nulla da ricalibrare -- se usata
    insieme a per_instance e' semplicemente un no-op innocuo (nn.InstanceNorm2d
    con track_running_stats=False ignora comunque l'aggiornamento delle
    running stats).

    Questa funzione fa girare il modello in train() mode su alcuni batch di
    training, SENZA calcolare gradienti ne' aggiornare i pesi (solo forward,
    no backward/optimizer.step) -- il solo effetto e' che InstanceNorm
    aggiorna le sue statistiche running_mean/running_var (comportamento
    automatico di nn.InstanceNorm2d con track_running_stats=True quando gira
    in train() mode), "rinfrescandole" sulla base dei pesi conv attuali.
    I coefficienti gamma/beta (se congelati da --freeze norm) NON vengono
    toccati, dato che non calcoliamo gradienti: solo le statistiche.
    """
    model.train()
    with torch.no_grad():
        for i, (imgs, segs) in enumerate(loader):
            if n_batches is not None and i >= n_batches:
                break
            imgs = imgs.to(device)
            _ = model(imgs)
    model.eval()


def train(args):
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'Device: {device}')

    # Fix random seed for reproducibility
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('Random seed fixed to 42')

    splits_path = os.path.join(args.data_dir, '..', 'splits_final.json')
    if os.path.exists(splits_path):
        train_cases, val_cases = load_splits(splits_path, fold=args.fold)
    else:
        patients = sorted(os.listdir(args.data_dir))
        all_cases = []
        for p in patients:
            pdir = os.path.join(args.data_dir, p)
            if not os.path.isdir(pdir):
                continue
            for f in os.listdir(pdir):
                if f.endswith('.nii.gz') and '_gt' not in f and '_4d' not in f:
                    frame = f.replace(p + '_frame', '').replace('.nii.gz', '')
                    all_cases.append(f'{p}_frame{frame}')
        n = len(all_cases)
        train_cases = all_cases[:int(n * 0.8)]
        val_cases   = all_cases[int(n * 0.8):]

    print(f'Train cases: {len(train_cases)}, Val cases: {len(val_cases)}')

    train_ds = ACDCDataset(args.data_dir, train_cases,
                           patch_size=(256, 224), augment=True)
    val_ds   = ACDCDataset(args.data_dir, val_cases,
                           patch_size=(256, 224), augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    print(f'Train slices: {len(train_ds)}, Val slices: {len(val_ds)}')

    clamp_values = None
    if args.clamp_values_json:
        with open(args.clamp_values_json) as f:
            clamp_values = json.load(f)
        print(f'Soglie di clamp calibrate caricate da: {args.clamp_values_json}')
        print(f'  ({len(clamp_values)} layer con soglia personalizzata)')

    act_overrides = None
    if args.degree1_layers:
        layer_names = [n.strip() for n in args.degree1_layers.split(',') if n.strip()]
        act_overrides = {name: 'linear' for name in layer_names}
        print(f'Attivazione forzata a grado 1 (linear, ax+b) nei layer: {layer_names}')
        print('  (tutti gli altri layer restano act_type={} come da --act)'.format(args.act))

    if args.soft_max_a is not None:
        print(f'Gating morbido attivo su PolyAct: soft_max_a={args.soft_max_a}, '
              f'soft_sharpness={args.soft_sharpness}')
        print('  (ogni layer si comporta quadratico pieno se |a| resta sotto la soglia,')
        print('   si "spegne" gradualmente verso lineare se |a| la supera)')

    if args.skip_mode == 'sum':
        print('skip_mode=sum: le skip connection vengono SOMMATE invece di concatenate')
        print('  (elimina lo squilibrio di gradiente upsampling/skip misurato con')
        print('   crypto/analyze_decoder_gradients.py -- ATTENZIONE: incompatibile con')
        print('   checkpoint allenati con skip_mode=concat, richiede training da zero)')

    if args.weight_standardization:
        print('weight_standardization=True: le Conv2d normali (non le ConvTranspose2d')
        print('  di upsampling) standardizzano i propri pesi ad ogni forward -- nessun')
        print('  costo aggiuntivo in HE (trasformazione sui pesi, non sulle attivazioni).')
        print('  Compatibile con checkpoint esistenti (stessa forma dei pesi).')

    model = HEFriendlyUNet(
        in_channels=1,
        num_classes=4,
        act_type=args.act,
        norm_type=args.norm,
        clamp_values=clamp_values,
        norm_mode=args.norm_mode,
        max_a_poly=args.max_a_poly,
        act_overrides=act_overrides,
        soft_max_a=args.soft_max_a,
        soft_sharpness=args.soft_sharpness,
        skip_mode=args.skip_mode,
        weight_standardization=args.weight_standardization,
        filters=args.filters,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model: act={args.act}, norm={args.norm}, norm_mode={args.norm_mode}, '
          f'skip_mode={args.skip_mode}, params={n_params:,}')

    # --- Caricamento pesi iniziali ---
    # --pretrained: solo le Conv2d dal baseline nnU-Net (usato in Fase II,
    #               partendo da un modello HEFriendlyUNet non ancora addestrato)
    # --init_from:  l'intero modello da un checkpoint di una fase precedente
    #               del training a fasi (es. Fase II -> Fase III)
    if args.pretrained:
        model = load_pretrained_conv(model, args.pretrained)
        print(f'Pesi conv caricati da: {args.pretrained}')
    if args.init_from:
        state = torch.load(args.init_from, map_location=device, weights_only=False)
        # load_checkpoint_shape_safe: come strict=False, ma gestisce anche
        # chiavi con nome uguale e forma diversa (es. PolyAct.a scalare vs
        # LinearAct.a shape [1] quando --degree1_layers cambia il tipo di
        # attivazione in alcuni blocchi, oppure l'intero decoder se
        # skip_mode differisce tra checkpoint e modello attuale) --
        # strict=False da solo NON basta in questi casi, solleva comunque
        # RuntimeError.
        n_loaded, skipped_shape, skipped_missing, kept_at_init = \
            load_checkpoint_shape_safe(model, state)
        print(f'  {n_loaded} parametri caricati dal checkpoint')
        if skipped_shape:
            print(f'  \u26a0\ufe0f  {len(skipped_shape)} parametri saltati per forma incompatibile '
                  f'(restano al valore di inizializzazione):')
            for name, ckpt_shape, model_shape in skipped_shape:
                print(f'      {name}: checkpoint {ckpt_shape} vs modello {model_shape}')
        print(f'  ({len(skipped_missing)} buffer/chiavi senza posto nel modello attuale ignorati, '
              f'attesi con norm_mode=per_instance)')
        print(f'Modello inizializzato da: {args.init_from}')
    if args.weight_standardization:
        from models.he_friendly import calibrate_ws_gain
        calibrate_ws_gain(model)
        print('WS gain calibrato sui pesi caricati (evita shock iniziale)')

        # --- Reset esplicito del coefficiente 'a' di ogni PolyAct ---
        # Necessario per un confronto sperimentale pulito tra "vincolo
        # strutturale (max_a_poly)" e "reset del punto di partenza".
        #
        # Con max_a_poly impostato, 'a' e' salvato come 'raw_a' nello state
        # dict; un checkpoint precedente (allenato con 'a' libero) non ha
        # quella chiave, quindi il caricamento strict=False la lascia gia'
        # al valore di inizializzazione (0.1) -- reset implicito, automatico.
        #
        # Con max_a_poly=None (modalita' libera, comportamento originale),
        # invece, la chiave si chiama 'a' in entrambi i casi: il checkpoint
        # la sovrascrive regolarmente con il valore fine-tunato (es. 1.11
        # in dec1.block.5), NESSUN reset avviene automaticamente. Senza
        # questo flag esplicito, un confronto "libero vs vincolato" sarebbe
        # confuso da due variabili diverse (vincolo E punto di partenza),
        # non isolerebbe l'effetto del vincolo da solo.
        if args.reset_poly_a:
            from models.he_friendly import PolyAct
            n_reset = 0
            with torch.no_grad():
                for m in model.modules():
                    if isinstance(m, PolyAct) and m.max_a is None:
                        m.a.fill_(0.1)
                        n_reset += 1
            print(f'  Reset esplicito di \'a\'=0.1 su {n_reset} layer PolyAct '
                  f'(--reset_poly_a attivo; no-op per i layer gia\' vincolati con max_a_poly)')

    # --- Freeze selettivo per il training a fasi ---
    trainable_params = apply_freeze(model, args.freeze)

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr,
                                  weight_decay=args.weight_decay)

    start_epoch = 1
    best_dice = 0.0
    history   = []
    patience_counter = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        best_dice = ckpt['best_dice']
        history = ckpt['history']
        start_epoch = ckpt['epoch'] + 1
        print(f'Resumed from epoch {ckpt["epoch"]}, best dice {best_dice:.3f}')

    def get_lr(epoch):
        if args.lr_schedule == 'constant':
            return args.lr
        elif args.lr_schedule == 'cosine':
            # Warmup lineare (se args.warmup > 0), poi decadimento coseno
            # fino a lr_min all'ultima epoca.
            if args.warmup > 0 and epoch <= args.warmup:
                return args.lr * epoch / args.warmup
            total = max(1, args.epochs - args.warmup)
            progress = (epoch - args.warmup) / total
            progress = min(max(progress, 0.0), 1.0)
            return args.lr_min + 0.5 * (args.lr - args.lr_min) * (1 + math.cos(math.pi * progress))
        elif args.lr_schedule == 'step':
            # Dimezza (o moltiplica per lr_decay) il lr ogni lr_step_size epoche
            n_decays = (epoch - 1) // args.lr_step_size
            return max(args.lr * (args.lr_decay ** n_decays), args.lr_min)
        else:
            raise ValueError(f'lr_schedule sconosciuto: {args.lr_schedule}')

    criterion = DiceCELoss(num_classes=4)

    run_name = f'act={args.act}_norm={args.norm}_mode={args.norm_mode}_skip-{args.skip_mode}_bs{args.batch_size}_lr{args.lr}'
    if args.lr_schedule != 'constant':
        run_name += f'_sched-{args.lr_schedule}'
    if args.freeze:
        run_name += f'_freeze-{args.freeze.replace(",", "-")}'
    if args.warmup > 0:
        run_name += f'_warmup{args.warmup}'
    if args.soft_max_a is not None:
        run_name += f'_softmaxa{args.soft_max_a}'
    if args.weight_standardization:
        run_name += '_ws'
    out_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    diagnostics_history = []
    diagnostics_path = None
    instance_norm_min_var = {}
    instance_norm_max_cv = {}
    instance_norm_spatial = {}
    if args.diagnostics_out:
        diagnostics_path = args.diagnostics_out
        print(f'Diagnostica drift PolyAct attiva, verra\' salvata in: {diagnostics_path}')
        # Attivi per tutta la durata del training. Due ipotesi testate:
        # (1) varianza quasi nulla in una singola istanza -- ESCLUSA dal
        #     trial diag2 (min_var sempre ~1e-3, mai vicina a eps=1e-5);
        # (2) stima di varianza rumorosa nei layer a bassa risoluzione
        #     spaziale (bottleneck/decoder profondo), misurata come
        #     coefficiente di variazione (CV) della varianza tra istanze
        #     dello stesso batch -- vedi register_instance_norm_variance_hooks().
        register_instance_norm_variance_hooks(
            model, instance_norm_min_var, instance_norm_max_cv, instance_norm_spatial)
        print('Hook di monitoraggio varianza InstanceNorm registrati.')

    for epoch in range(start_epoch, args.epochs + 1):
        # Aggiorna lr
        current_lr = get_lr(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        instance_norm_min_var.clear()
        instance_norm_max_cv.clear()
        # instance_norm_spatial NON va azzerato: e' statico per l'architettura.

        model.train()
        train_loss = 0.0
        n_clamped_train = 0
        epoch_penalty = 0.0
        for imgs, segs in tqdm(train_loader,
                               desc=f'Epoch {epoch}/{args.epochs} [train]',
                               leave=False):
            imgs = imgs.to(device)
            segs = segs.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            logits, n_clamp = safe_clamp_logits(logits)
            n_clamped_train += n_clamp
            loss = criterion(logits, segs)
            if args.act_penalty_weight > 0:
                pen = activation_penalty(model, penalty_threshold=args.act_penalty_threshold,
                                        use_layer_threshold=args.act_penalty_use_layer_threshold)
                loss = loss + args.act_penalty_weight * pen
                epoch_penalty += pen.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        if args.act_penalty_weight > 0 and len(train_loader) > 0:
            print(f'  Penalita\' attivazioni media (training): {epoch_penalty/len(train_loader):.4f}')
        if n_clamped_train > 0:
            print(f'  \u26a0\ufe0f  Clamp attivato {n_clamped_train} volte nei logits di training')

        # --- Diagnostica drift PolyAct (fine epoca, PRIMA della validazione) ---
        # L'ordine e' importante: .last_raw deve ancora riferirsi all'ultimo
        # batch di TRAINING, non a uno di validazione che lo sovrascriverebbe.
        if diagnostics_path:
            diag = collect_polyact_diagnostics(model)
            diag['epoch'] = epoch
            # Snapshot della varianza minima vista in ogni InstanceNorm2d
            # durante TUTTO il training loop di questa epoca (non solo
            # l'ultimo batch, a differenza delle statistiche PolyAct sopra --
            # qui vogliamo catturare anche un singolo batch "cattivo"
            # capitato a meta' epoca).
            diag['instance_norm_min_var'] = dict(instance_norm_min_var)
            diag['instance_norm_max_cv'] = dict(instance_norm_max_cv)
            diag['instance_norm_spatial_hw'] = dict(instance_norm_spatial)
            diagnostics_history.append(diag)
            with open(diagnostics_path, 'w') as f:
                json.dump(diagnostics_history, f, indent=2)

            if diag['polyact']:
                max_a = max(abs(v['a']) for v in diag['polyact'].values())
                max_a_layer = max(diag['polyact'].items(), key=lambda kv: abs(kv[1]['a']))[0]
                max_p99 = max((v.get('raw_p99_abs') or 0) for v in diag['polyact'].values())
                max_p99_layer = max(diag['polyact'].items(), key=lambda kv: (kv[1].get('raw_p99_abs') or 0))[0]
                max_abs = max((v.get('raw_max_abs') or 0) for v in diag['polyact'].values())
                max_abs_layer = max(diag['polyact'].items(), key=lambda kv: (kv[1].get('raw_max_abs') or 0))[0]

                print(f'  [diag] conv_weight_norm={diag["conv_weight_norm"]:.2f}  '
                      f'max|a|={max_a:.4f} ({max_a_layer})  '
                      f'max_raw_p99={max_p99:.2f} ({max_p99_layer})  '
                      f'max_raw_ABS={max_abs:.2f} ({max_abs_layer})')

                if args.soft_max_a is not None:
                    gates = {k: v.get('gate', 1.0) for k, v in diag['polyact'].items()}
                    min_gate_layer = min(gates.items(), key=lambda kv: kv[1])
                    n_gated = sum(1 for g in gates.values() if g < 0.9)
                    print(f'  [diag] gate minimo: {min_gate_layer[1]:.4f} ({min_gate_layer[0]})  '
                          f'-- {n_gated}/{len(gates)} layer con gate<0.9 (parzialmente "spenti")')
            else:
                # Nessuna PolyAct nel modello (es. --act linear su tutta la
                # rete): non c'e' nulla da riportare su 'a'/raw pre-clamp,
                # ma conv_weight_norm resta comunque utile da monitorare.
                print(f'  [diag] conv_weight_norm={diag["conv_weight_norm"]:.2f}  '
                      f'(nessuna PolyAct nel modello, --act={args.act})')

            if instance_norm_min_var:
                min_var_layer = min(instance_norm_min_var.items(), key=lambda kv: kv[1])
                print(f'  [diag] varianza minima InstanceNorm in tutta l\'epoca: '
                      f'{min_var_layer[1]:.2e} (layer {min_var_layer[0]})  '
                      f'[eps di default = 1e-5]')

            if instance_norm_max_cv:
                top_cv = sorted(instance_norm_max_cv.items(), key=lambda kv: kv[1], reverse=True)[:5]
                print('  [diag] Top 5 layer per dispersione (coeff. di variazione) della varianza tra istanze:')
                for name, cv in top_cv:
                    hw = instance_norm_spatial.get(name, '?')
                    print(f'      {name:20s} cv={cv:7.2f}   pixel/istanza={hw}')

        # --- Ricalibrazione periodica delle statistiche InstanceNorm ---
        # (no-op se --norm_mode per_instance, vedi docstring della funzione)
        if args.recalibrate_every > 0 and epoch % args.recalibrate_every == 0:
            recalibrate_norm_stats(model, train_loader, device, n_batches=args.recalibrate_batches)
            n_batches_str = str(args.recalibrate_batches) if args.recalibrate_batches else 'tutti i'
            print(f'  Statistiche InstanceNorm ricalibrate ({n_batches_str} batch di training, pesi conv attuali)')

        # ---------------------------------------------------------------
        # VALIDAZIONE con diagnostica NaN/Inf nei logits
        # ---------------------------------------------------------------
        model.eval()
        val_loss = 0.0
        n_valid_batches = 0
        n_clamped_val = 0
        dice_rv, dice_myo, dice_lv = [], [], []
        nan_cases_this_epoch = []

        with torch.no_grad():
            for batch_idx, (imgs, segs) in enumerate(val_loader):
                imgs = imgs.to(device)
                segs = segs.to(device)
                logits = model(imgs)

                # --- DIAGNOSTICA: quali sample nel batch hanno logits non finiti ---
                # NB: ACDCDataset ha una entry per slice 2D (non per paziente/frame),
                # quindi il mapping corretto passa da val_ds.slices, non da val_cases.
                bad_mask = ~torch.isfinite(logits).all(dim=(1, 2, 3))
                if bad_mask.any():
                    start = batch_idx * val_loader.batch_size
                    bad_local_idxs = bad_mask.nonzero(as_tuple=True)[0].tolist()
                    for li in bad_local_idxs:
                        global_idx = start + li
                        if global_idx < len(val_ds.slices):
                            img_path, _, slice_idx = val_ds.slices[global_idx]
                            # img_path tipo '.../patient002/patient002_frame01.nii.gz'
                            fname = os.path.basename(img_path).replace('.nii.gz', '')
                            case_id = f'{fname}_slice{slice_idx}'
                        else:
                            case_id = f'idx_{global_idx}'
                        nan_cases_this_epoch.append(case_id)

                # --- CLAMP di sicurezza sui logits, prima della loss ---
                logits, n_clamp = safe_clamp_logits(logits)
                n_clamped_val += n_clamp

                loss = criterion(logits, segs)
                if torch.isfinite(loss):
                    val_loss += loss.item()
                    n_valid_batches += 1

                preds = logits.argmax(dim=1)
                scores = dice_score(preds, segs)
                dice_rv.append(scores[1])
                dice_myo.append(scores[2])
                dice_lv.append(scores[3])

        val_loss = val_loss / n_valid_batches if n_valid_batches > 0 else 999.0

        if nan_cases_this_epoch:
            unique_cases = sorted(set(nan_cases_this_epoch))
            print(f'  \u26a0\ufe0f  NaN/Inf nei logits (prima del clamp) — {len(unique_cases)} casi: {unique_cases[:10]}'
                  f'{" ..." if len(unique_cases) > 10 else ""}')
        if n_clamped_val > 0:
            print(f'  \u26a0\ufe0f  Clamp attivato {n_clamped_val} volte nei logits di validazione')

        rv  = sum(dice_rv)  / len(dice_rv)
        myo = sum(dice_myo) / len(dice_myo)
        lv  = sum(dice_lv)  / len(dice_lv)
        mean_dice = (rv + myo + lv) / 3

        print(f'Epoch {epoch:3d} | loss {train_loss:.4f} | val_loss {val_loss:.4f} | '
              f'RV {rv:.3f} MYO {myo:.3f} LV {lv:.3f} | mean {mean_dice:.3f} | lr {current_lr:.2e}')

        history.append({
            'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
            'dice_rv': rv, 'dice_myo': myo, 'dice_lv': lv,
            'mean_dice': mean_dice, 'lr': current_lr,
            'nan_cases': sorted(set(nan_cases_this_epoch))
        })

        if mean_dice > best_dice:
            best_dice = mean_dice
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(out_dir, 'best_model.pth'))
            print(f'  → saved best model (mean dice {best_dice:.3f})')
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop:
                print(f'\nEarly stopping at epoch {epoch}')
                break

        if epoch % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'history': history,
            }, os.path.join(out_dir, f'checkpoint_epoch{epoch}.pth'))
            print(f'  → checkpoint saved at epoch {epoch}')


    torch.save(model.state_dict(), os.path.join(out_dir, 'final_model.pth'))
    with open(os.path.join(out_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f'\nDone. Best mean Dice: {best_dice:.3f}')
    print(f'Results saved in: {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=os.path.expanduser('~/Desktop/tesi_acdc/training'))
    parser.add_argument('--out_dir',  default='results')
    parser.add_argument('--act',      default='poly', choices=['identity', 'linear', 'squared', 'poly'])
    parser.add_argument('--norm',     default='none', choices=['none', 'batch', 'instance', 'group', 'poly'])
    parser.add_argument('--norm_mode', default='population', choices=['population', 'per_instance'],
                        help="Rilevante solo con --norm instance. 'population' (default, invariato): "
                             "InstanceNorm usa statistiche di popolazione fisse (track_running_stats=True), "
                             "e' il comportamento originale con cui sono stati allenati i checkpoint "
                             "Fase I/II/III finora. 'per_instance': statistiche calcolate live per ogni "
                             "immagine, sia in train che in eval (track_running_stats=False) -- vedi "
                             "crypto/check_per_instance_norm.py per la motivazione e i risultati "
                             "(risolve l'esplosione numerica alla radice e migliora il Dice).")
    parser.add_argument('--skip_mode', default='concat', choices=['concat', 'sum'],
                        help="'concat' (default, invariato): le skip connection vengono concatenate "
                             "lungo i canali, che vengono raddoppiati prima di ogni ConvBlock del "
                             "decoder -- comportamento originale. 'sum': le skip connection vengono "
                             "SOMMATE invece di concatenate. Motivazione (crypto/analyze_decoder_"
                             "gradients.py): con 'concat' il gradiente che arriva al punto di unione "
                             "dal ramo di upsampling e quello dalla skip possono differire fino a "
                             "~14:1 nell'ultimo stage del decoder -- con 'sum', per costruzione "
                             "matematica, i due contributi ricevono sempre lo stesso gradiente. "
                             "Bonus: dato che i canali di upsampling e skip coincidono gia' ad ogni "
                             "stage, 'sum' non raddoppia i canali come 'concat' -- meno parametri nel "
                             "decoder. ATTENZIONE: un checkpoint allenato con 'concat' non e' "
                             "compatibile con un modello 'sum' (shape diverse nella prima Conv2d di "
                             "ogni blocco decoder) -- richiede training da zero, non caricabile via "
                             "--init_from/--pretrained da un checkpoint 'concat' esistente.")
    parser.add_argument('--max_a_poly', type=float, default=None,
                        help="Se omesso (default, invariato): il coefficiente 'a' (termine quadratico) "
                             "di ogni PolyAct resta un parametro libero, comportamento originale. Se "
                             "specificato (es. 1.0): 'a' viene vincolato strutturalmente in "
                             "(-max_a_poly, +max_a_poly) tramite riparametrizzazione tanh -- impedisce "
                             "all'ottimizzatore di far crescere il coefficiente quadratico oltre un "
                             "tetto fisso, attaccando l'effetto valanga (ax^2+bx+c non limitata) alla "
                             "radice invece di limitarsi a tagliare l'output dopo il fatto (clamp/STE). "
                             "Vedi PolyAct.current_a() in models/he_friendly.py.")
    parser.add_argument('--soft_max_a', type=float, default=None,
                        help="Se omesso (default, invariato): nessun gating morbido, comportamento "
                             "originale. Se specificato (es. 0.2): ogni PolyAct applica un fattore di "
                             "gate (tra 0 e 1, sigmoid) al proprio termine quadratico -- vicino a 1 "
                             "(quadratico pieno) se |a| resta sotto soft_max_a, vicino a 0 (comportamento "
                             "lineare, bx+c) se |a| lo supera. A differenza di --max_a_poly (vincolo "
                             "rigido strutturale, sempre attivo) e di --degree1_layers (commutazione "
                             "netta e permanente a priori su blocchi scelti), qui la transizione e' "
                             "MORBIDA e derivabile ovunque (nessuna discontinuita' nel gradiente), e "
                             "avviene layer per layer durante il training in base a come si comporta "
                             "ciascuno, non decisa in anticipo su quali blocchi. Richiesta di Aurora, "
                             "dopo la scoperta dello squilibrio nei gradienti skip/upsampling (vedi "
                             "--skip_mode) -- l'idea e' lasciare che PolyAct 'si auto-limiti' solo dove "
                             "e quando serve davvero. Vedi PolyAct.gate_value() in models/he_friendly.py.")
    parser.add_argument('--soft_sharpness', type=float, default=10.0,
                        help="Controlla quanto e' brusca la transizione del gating morbido (solo se "
                             "--soft_max_a e' specificato, altrimenti ignorato). Valori alti (20-50): "
                             "transizione quasi netta ma ancora derivabile. Valori bassi (2-5): "
                             "transizione molto graduale. Default 10.0.")
    parser.add_argument('--weight_standardization', action='store_true',
                        help="Se presente (default: disattivato, comportamento originale invariato), "
                             "ogni Conv2d 'normale' della rete (dentro i ConvBlock di encoder/decoder, "
                             "e il layer di output finale -- NON le ConvTranspose2d di upsampling) "
                             "standardizza i propri pesi (media 0, std 1 per canale di output) ad ogni "
                             "forward, prima di usarli nella convoluzione. Trasformazione sui PARAMETRI, "
                             "non sulle attivazioni -- zero costo aggiuntivo in HE (si applica una volta "
                             "in chiaro, prima di cifrare). Motivazione: la diagnostica (conv_weight_norm) "
                             "ha mostrato la norma dei pesi crescere nei run poi collassati -- WS vincola "
                             "strutturalmente la scala di ogni filtro. Compatibile con checkpoint "
                             "esistenti (stessa forma dei pesi, comportamento numerico diverso). Vedi "
                             "WSConv2d in models/he_friendly.py.")
    parser.add_argument('--reset_poly_a', action='store_true',
                        help="Se presente, forza 'a'=0.1 su ogni PolyAct DOPO aver caricato --init_from, "
                             "anche in modalita' libera (max_a_poly=None), dove altrimenti il checkpoint "
                             "lo sovrascriverebbe col valore fine-tunato. Serve per isolare l'effetto del "
                             "vincolo strutturale (max_a_poly) da quello del semplice reset del punto di "
                             "partenza in un confronto sperimentale pulito. Default: disattivato "
                             "(comportamento originale invariato).")
    parser.add_argument('--degree1_layers', default=None,
                        help="Lista CSV di nomi di blocchi (es. 'enc4,enc5,dec4,dec3') in cui forzare "
                             "un'attivazione di GRADO 1 (LinearAct, ax+b) invece di PolyAct (grado 2, "
                             "ax^2+bx+c), rimuovendo il termine quadratico esattamente in quei layer. "
                             "Tutti gli altri layer non elencati restano act_type=poly come specificato "
                             "da --act. Default None (comportamento originale invariato, nessun override). "
                             "Motivazione: la diagnostica (crypto/analyze_polyact_drift.py) mostra che "
                             "vincolare il coefficiente 'a' (--max_a_poly) non basta a fermare la crescita "
                             "del clamp nei layer a bassa risoluzione spaziale (bottleneck) -- rimuovere "
                             "il quadratico alla radice solo li' attacca il meccanismo direttamente "
                             "(proposta 3 di Aurora, applicata chirurgicamente ai layer fragili).")
    parser.add_argument('--epochs',   type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr',       type=float, default=1e-4)
    parser.add_argument('--lr_schedule', default='constant', choices=['constant', 'cosine', 'step'],
                        help="'constant' (default, comportamento originale): lr fisso per tutto il training. "
                             "'cosine': warmup lineare (se --warmup>0) poi decadimento a coseno fino a lr_min. "
                             "'step': dimezza (o *lr_decay) il lr ogni lr_step_size epoche.")
    parser.add_argument('--lr_min',   type=float, default=1e-6,
                        help='lr minimo raggiungibile con schedule cosine/step')
    parser.add_argument('--lr_step_size', type=int, default=30,
                        help='ogni quante epoche decade il lr con schedule step')
    parser.add_argument('--lr_decay', type=float, default=0.5,
                        help='fattore moltiplicativo del decadimento con schedule step')
    parser.add_argument('--fold',     type=int, default=0)
    parser.add_argument('--recalibrate_every', type=int, default=0,
                        help='Ogni quante epoche ricalibrare le statistiche InstanceNorm sui pesi '
                             'conv attuali (0 = disattivato, default, statistiche completamente '
                             'fisse come nel metodo originale). Utile in Fase III con --norm_mode '
                             'population, dove i pesi conv sbloccati si allontanano da quelli su cui '
                             'IN era stata calibrata. No-op se --norm_mode per_instance.')
    parser.add_argument('--recalibrate_batches', type=int, default=None,
                        help='Numero di batch di training da usare per la ricalibrazione IN '
                             '(default None = un giro completo del training set, piu\' lento ma piu\' stabile)')
    parser.add_argument('--act_penalty_weight', type=float, default=0.0,
                        help='Peso della penalita\' esplicita sulle attivazioni PolyAct '
                             '(proposta Aurora, alternativa allo straight-through estimator). '
                             '0.0 = disattivata (default, comportamento invariato). '
                             'Penalizza quanto il valore grezzo pre-clamp supera '
                             '--act_penalty_threshold, spingendo la rete a restare naturalmente '
                             'contenuta.')
    parser.add_argument('--act_penalty_threshold', type=float, default=50.0,
                        help='Soglia GLOBALE di riferimento per activation_penalty, usata solo se '
                             '--act_penalty_use_layer_threshold NON e\' specificato. Indipendente dal '
                             'clamp_value di ciascuna PolyAct/--clamp_values_json -- permette di '
                             'disattivare il clamp (soglie enormi via --clamp_values_json) mantenendo '
                             'la penalita\' attiva con la sua soglia originale, per testare se la '
                             'penalita\' da sola basta a controllare le attivazioni.')
    parser.add_argument('--act_penalty_use_layer_threshold', action='store_true',
                        help='Se presente, la penalita\' usa la soglia CALIBRATA PER-LAYER di ciascuna '
                             'PolyAct (m.clamp_value, la stessa usata dal clamp interno/--clamp_values_json) '
                             'invece del valore globale --act_penalty_threshold. Allinea la penalita\' al '
                             'vincolo che il clamp sta gia\' imponendo layer per layer -- senza questo, '
                             'un layer con soglia di clamp stretta (es. 7.0) viene limitato a forza dal '
                             'clamp ma la penalita\' non se ne accorge fino a superare 50 (molto piu\' '
                             'largo), lasciando che i pesi continuino a spingere verso valori grandi che '
                             'si accumulano nei layer a valle. NON compatibile con test tipo "penalita\' '
                             'senza clamp" (in quel caso disattivare questo flag).')
    parser.add_argument('--clamp_values_json', default=None,
                        help='Path a un file JSON con soglie di clamp calibrate per layer '
                             '(prodotto da crypto/calibrate_clamp_threshold.py). '
                             'Se omesso, tutte le PolyAct usano il default (50.0)')
    parser.add_argument('--pretrained', default=None,
                        help='Path a pesi conv-only del baseline (usato in Fase II)')
    parser.add_argument('--init_from', default=None,
                        help='Path a un checkpoint completo di una fase precedente (usato in Fase III)')
    parser.add_argument('--freeze', default=None,
                        help="Gruppi di parametri da congelare, CSV tra 'conv','act','norm'. "
                             "Es: --freeze conv (Fase II, congela le conv, allena act+norm) "
                             "oppure --freeze norm (Fase III, congela IN, allena conv+act)")
    parser.add_argument('--early_stop', type=int, default=20)
    parser.add_argument('--warmup',   type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--diagnostics_out', default=None,
                        help='Path JSON dove salvare, epoca per epoca, i coefficienti (a,b,c) e le '
                             'statistiche del valore pre-clamp (last_raw) di ogni PolyAct, piu\' la '
                             'norma complessiva dei pesi conv. None (default) = diagnostica disattivata, '
                             'nessun effetto su training/performance. Usalo per capire QUALE layer '
                             'inizia a derivare prima di un collasso improvviso del Dice.')
    parser.add_argument('--filters', type=int, nargs=6, default=None,
                        help="6 interi [enc0..enc5], canali per stage. Default None "
                            "(dimensionamento originale [32,64,128,256,512,512]). "
                            "Cambia le shape dei pesi -- richiede training da zero, "
                            "non compatibile con --init_from/--pretrained esistenti.")
    args = parser.parse_args()
    train(args)