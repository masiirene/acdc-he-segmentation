import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

class IdentityAct(nn.Module):
    """x — linear activation, HE depth cost: 0"""
    def forward(self, x):
        return x


class LinearAct(nn.Module):
    """ax — learnable scalar, HE depth cost: 0"""
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return self.a * x


class SquaredAct(nn.Module):
    """x² — simplest nonlinear polynomial, HE depth cost: 1"""
    def forward(self, x):
        return x * x


class PolyAct(nn.Module):
    """ax² + bx + c — learnable polynomial, HE depth cost: 1
    Inspired by ULD-Net (Xie et al., ICLR 2026).
    Initialization from ULD-Net: c0=0.5, c1=1, c2=0.1

    NB: ax²+bx+c non e' una funzione limitata. Con a≈0.1, appena |x| supera
    circa 1/a=10 il termine quadratico inizia a dominare e amplifica il
    valore ad ogni layer successivo. In una rete profonda (qui: 22 istanze
    di PolyAct in cascata tra encoder e decoder) questo puo' portare a
    un'esplosione esponenziale da valori "normali" (~10-40) fino a Inf/NaN
    nel giro di pochi stage — osservato empiricamente su alcuni pazienti
    ACDC (vedi inspect_activations.py).

    STRAIGHT-THROUGH ESTIMATOR (suggerimento Aurora, dopo che il clamp
    "semplice" si e' rivelato indispensabile anche in inferenza, non solo
    in training -- vedi crypto/check_inference_stability.py):
    Un clamp normale ha derivata zero nei punti clampati: durante il
    backward, la rete "non vede" l'errore proprio dove servirebbe di piu'
    correggersi, rendendo difficile imparare coefficienti che non abbiano
    bisogno del clamp. Con lo straight-through estimator, il FORWARD resta
    clampato (stabilita' numerica preservata, identico a prima), ma il
    BACKWARD usa il gradiente pieno del polinomio non clampato -- cosi' la
    rete riceve un segnale di correzione anche nei punti critici, ed e'
    incentivata a imparare pesi/coefficienti che tengano i valori dentro
    il range da sola, riducendo nel tempo quanto il clamp deve intervenire.

    VINCOLO STRUTTURALE SU 'a' (max_a, evoluzione della diagnosi di drift):
    Il clamp e lo STE limitano l'OUTPUT dopo che il polinomio l'ha gia'
    prodotto -- curano il sintomo. La diagnostica (crypto/analyze_polyact_
    drift.py) mostra che nei layer a bassa risoluzione spaziale (bottleneck
    enc5, primi stage del decoder dec4/dec3 -- 56-224 pixel per istanza) il
    coefficiente 'a' e' gia' grande (0.78-0.90) e la stima di varianza
    per-istanza e' statisticamente rumorosa su cosi' pochi campioni: un
    salto anche piccolo di 'a' amplifica quel rumore in modo quadratico,
    producendo i collassi improvvisi osservati (sensibili persino a
    differenze numeriche impercettibili tra run "identici" su MPS).

    Se max_a e' specificato (non None), 'a' viene riparametrizzato con una
    tanh: a = max_a * tanh(raw_a), dove raw_a e' il parametro libero
    allenato dall'ottimizzatore. Per costruzione matematica, |a| < max_a
    SEMPRE, qualunque gradiente l'ottimizzatore proponga -- non e' un
    clamp applicato a posteriori, e' un limite sul meccanismo stesso
    dell'effetto valanga. Se max_a e' None (default), 'a' resta un
    parametro libero come nel comportamento originale, invariato.
    """
    def __init__(self, clamp_value: float = 50.0, max_a: float = None):
        super().__init__()
        self.max_a = max_a
        if max_a is None:
            # Comportamento originale, invariato: 'a' e' un parametro libero.
            self.a = nn.Parameter(torch.tensor(0.1))
        else:
            # Riparametrizzazione vincolata: raw_a e' libero, ma a=max_a*tanh(raw_a)
            # e' strutturalmente confinato in (-max_a, +max_a).
            init_a = min(0.1, max_a * 0.99)
            raw_init = math.atanh(init_a / max_a)
            self.raw_a = nn.Parameter(torch.tensor(raw_init))
        self.b = nn.Parameter(torch.tensor(1.0))   # c1
        self.c = nn.Parameter(torch.tensor(0.5))   # c0
        self.clamp_value = clamp_value

    def current_a(self):
        """Valore effettivo di 'a' da usare nel forward e nella diagnostica,
        sia in modalita' libera che vincolata."""
        if self.max_a is None:
            return self.a
        return self.max_a * torch.tanh(self.raw_a)

    def forward(self, x):
        a = self.current_a()
        out = a * x * x + self.b * x + self.c
        # Esposto per un'eventuale penalita' esplicita nella loss (proposta
        # di Aurora, alternativa/complementare allo straight-through
        # estimator): il gradiente resta attaccato, cosi' train.py puo'
        # sommare un termine "quanto supero la soglia" alla loss principale,
        # spingendo la rete a non aver bisogno del clamp fin dall'inizio.
        self.last_raw = out
        clamped = torch.clamp(out, -self.clamp_value, self.clamp_value)
        # Straight-through: il valore restituito e' quello clampato, ma il
        # gradiente che si propaga indietro e' quello del polinomio pieno
        # (il termine (clamped-out) e' "staccato" dal grafo autograd).
        return out + (clamped - out).detach()


class PolyNorm(nn.Module):
    """
    PolyNorm stabile: usa 1/sqrt(var + eps) approssimato con clamp aggressivo.
    In inferenza usa running stats fisse -> compatibile HE.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta  = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var',  torch.ones(num_features))

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=[0, 2, 3])
            var  = x.var(dim=[0, 2, 3], unbiased=False)
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * var
        else:
            mean = self.running_mean
            var  = self.running_var

        mean = mean.view(1, -1, 1, 1)
        var  = var.view(1, -1, 1, 1)

        # Normalizza con var clampata — stabile sempre
        var_clamped = var.clamp(min=self.eps)
        x_norm = (x - mean) / torch.sqrt(var_clamped)

        # Approssimazione polinomiale di 1/sqrt in inferenza
        # In training usiamo sqrt vera per stabilità
        # In inferenza: 1/sqrt(v) ≈ costante precalcolata * (1 - 0.5*(v-1))
        # per v vicino a 1 (dopo normalizzazione running)

        gamma = self.gamma.view(1, -1, 1, 1)
        beta  = self.beta.view(1, -1, 1, 1)
        return gamma * x_norm + beta


ACTIVATIONS = {
    'identity': IdentityAct,
    'linear':   LinearAct,
    'squared':  SquaredAct,
    'poly':     PolyAct,
}


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def get_norm(norm_type: str, num_features: int, norm_mode: str = 'population'):
    """
    norm_mode (rilevante solo per norm_type='instance'):
      - 'population' (default): InstanceNorm2d con track_running_stats=True,
        cioe' usa una media/varianza fissa calcolata durante il training e
        congelata in inferenza -- comportamento ORIGINALE, invariato, e'
        quello con cui sono stati allenati tutti i checkpoint finora
        (Fase I/II/III). Compatibile all'indietro.
      - 'per_instance': InstanceNorm2d con track_running_stats=False, cioe'
        calcola sempre media/varianza LIVE sulla singola immagine corrente,
        sia in train() che in eval() -- il comportamento "naturale" di
        InstanceNorm per design. Vedi crypto/check_per_instance_norm.py:
        risolve l'esplosione numerica alla radice ed e' naturalmente
        compatibile con HE (in inferenza reale il server riceve un
        paziente/ciphertext alla volta, quindi media/varianza per-istanza
        sono un'operazione lineare fattibile con EvalSumRows/Cols, non
        serve nessuna costante di popolazione precalcolata).
    """
    if norm_type == 'none':
        return nn.Identity()
    elif norm_type == 'batch':
        return nn.BatchNorm2d(num_features, affine=True)
    elif norm_type == 'poly':
        return PolyNorm(num_features)
    elif norm_type == 'instance':
        return nn.InstanceNorm2d(
            num_features, affine=True,
            track_running_stats=(norm_mode != 'per_instance'),
        )
    else:
        raise ValueError(f'Unknown norm_type: {norm_type}')


# ---------------------------------------------------------------------------
# Basic building block
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv3x3 → Norm → Act, repeated twice per stage (as in nnU-Net).

    clamp_values: coppia opzionale (soglia_act1, soglia_act2) per calibrare
    individualmente le due PolyAct di questo blocco (vedi
    crypto/calibrate_clamp_threshold.py). Se None, usa il default di
    PolyAct (50.0) per entrambe -- comportamento originale, invariato.

    norm_mode: 'population' (default, invariato) o 'per_instance' -- vedi
    get_norm() per i dettagli. Propagato identico a entrambe le Norm del
    blocco.
    """
    def __init__(self, in_ch, out_ch, stride=1,
                 norm_type='none', act_type='poly', clamp_values=None,
                 norm_mode='population', max_a=None):
        super().__init__()
        Act = ACTIVATIONS[act_type]

        def make_act():
            if act_type == 'poly':
                kwargs = {'max_a': max_a}
                if clamp_values is not None:
                    idx = make_act.counter
                    make_act.counter += 1
                    kwargs['clamp_value'] = clamp_values[idx]
                return Act(**kwargs)
            return Act()
        make_act.counter = 0

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=True),
            get_norm(norm_type, out_ch, norm_mode),
            make_act(),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=True),
            get_norm(norm_type, out_ch, norm_mode),
            make_act(),
        )

    def forward(self, x):
        return self.block(x)


# ---------------------------------------------------------------------------
# HE-friendly U-Net
# ---------------------------------------------------------------------------

class HEFriendlyUNet(nn.Module):
    """
    2D U-Net with the same structure as nnU-Net on ACDC:
      - 6 encoder stages, 5 decoder stages
      - Stride convolution instead of MaxPool (HE-compatible)
      - Configurable activation and normalization

    Args:
        in_channels:  number of input channels (1 for cine MRI)
        num_classes:  number of output classes (4 for ACDC)
        act_type:     'identity' | 'linear' | 'squared' | 'poly'
        norm_type:    'none' | 'batch' | 'instance'
        norm_mode:    'population' | 'per_instance' (vedi get_norm())
    """

    def __init__(self, in_channels=1, num_classes=4,
                 act_type='poly', norm_type='none', clamp_values=None,
                 norm_mode='population', max_a_poly=None, act_overrides=None):
        """
        clamp_values: dizionario opzionale {nome_layer: soglia}, con chiavi
        del tipo 'enc0.block.2', 'enc0.block.5', ecc. (lo stesso formato
        prodotto da crypto/calibrate_clamp_threshold.py). Se None, ogni
        PolyAct usa il default (50.0) -- comportamento originale invariato.

        norm_mode: 'population' (default, invariato -- comportamento
        originale con cui sono stati allenati i checkpoint Fase I/II/III)
        oppure 'per_instance' (statistiche InstanceNorm calcolate live per
        ogni immagine, sia in train che in eval -- vedi
        crypto/check_per_instance_norm.py per la motivazione e i risultati
        sperimentali). Rilevante solo se norm_type='instance'; assegnato
        PRIMA di costruire encoder/decoder perche' get_norm() ne ha
        bisogno durante la costruzione dei blocchi.

        max_a_poly: se None (default, invariato), il coefficiente 'a' di
        ogni PolyAct resta un parametro libero come nel comportamento
        originale. Se un float, 'a' viene vincolato strutturalmente in
        (-max_a_poly, +max_a_poly) tramite riparametrizzazione tanh (vedi
        PolyAct.current_a()) -- impedisce all'ottimizzatore di far
        crescere il coefficiente quadratico oltre un tetto fisso e sicuro,
        attaccando il meccanismo dell'effetto valanga alla radice invece
        di limitarsi a tagliare l'output dopo il fatto (clamp/STE).

        act_overrides: dizionario opzionale {nome_blocco: act_type}, es.
        {'enc4': 'linear', 'enc5': 'linear', 'dec4': 'linear', 'dec3': 'linear'}.
        Se un blocco compare qui, ENTRAMBE le sue attivazioni usano
        act_type indicato invece di quello globale -- permette di
        rimuovere il termine quadratico (grado 1, 'linear': ax+b invece di
        ax^2+bx+c) SOLO nei layer identificati come strutturalmente
        fragili (bassa risoluzione spaziale / bottleneck), lasciando
        PolyAct pieno (grado 2) dove la diagnostica mostra che e' sempre
        stato stabile. None (default, invariato): tutti i blocchi usano
        act_type globale, comportamento originale.

        Motivazione (vedi crypto/analyze_polyact_drift.py, trial
        --max_a_poly 0.11): vincolare il coefficiente 'a' anche in modo
        molto stringente (bloccato a ~0.10, praticamente congelato) NON
        ferma la crescita degli interventi di clamp -- segno che il
        problema non e' la deriva di 'a' in se', ma il fatto che QUALSIASI
        termine quadratico non nullo, combinato con input che si allargano
        naturalmente nei layer a bassa risoluzione (bottleneck), produce
        code pesanti che il clamp deve tagliare. Rimuovere il quadratico
        alla radice in quei layer specifici attacca il meccanismo, non il
        sintomo.
        """
        super().__init__()

        self.act_type  = act_type
        self.norm_type = norm_type
        self.norm_mode = norm_mode
        self.max_a_poly = max_a_poly
        self.act_overrides = act_overrides or {}

        def block_act_type(block_name):
            """Ritorna act_type globale, a meno che act_overrides non lo
            sovrascriva per QUESTO specifico blocco."""
            return self.act_overrides.get(block_name, act_type)

        def cv(block_name):
            """Estrae la coppia (soglia_act1, soglia_act2) per un blocco dal
            dizionario di calibrazione, o None se non disponibile/non fornito."""
            if clamp_values is None:
                return None
            k1, k2 = f'{block_name}.block.2', f'{block_name}.block.5'
            if k1 in clamp_values and k2 in clamp_values:
                return (clamp_values[k1], clamp_values[k2])
            return None

        filters = [32, 64, 128, 256, 512, 512]

        # Encoder
        self.enc0 = ConvBlock(in_channels, filters[0], stride=1,
                              norm_type=norm_type, act_type=block_act_type('enc0'), clamp_values=cv('enc0'),
                              norm_mode=norm_mode, max_a=max_a_poly)
        self.enc1 = ConvBlock(filters[0], filters[1], stride=2,
                              norm_type=norm_type, act_type=block_act_type('enc1'), clamp_values=cv('enc1'),
                              norm_mode=norm_mode, max_a=max_a_poly)
        self.enc2 = ConvBlock(filters[1], filters[2], stride=2,
                              norm_type=norm_type, act_type=block_act_type('enc2'), clamp_values=cv('enc2'),
                              norm_mode=norm_mode, max_a=max_a_poly)
        self.enc3 = ConvBlock(filters[2], filters[3], stride=2,
                              norm_type=norm_type, act_type=block_act_type('enc3'), clamp_values=cv('enc3'),
                              norm_mode=norm_mode, max_a=max_a_poly)
        self.enc4 = ConvBlock(filters[3], filters[4], stride=2,
                              norm_type=norm_type, act_type=block_act_type('enc4'), clamp_values=cv('enc4'),
                              norm_mode=norm_mode, max_a=max_a_poly)
        self.enc5 = ConvBlock(filters[4], filters[5], stride=2,
                              norm_type=norm_type, act_type=block_act_type('enc5'), clamp_values=cv('enc5'),
                              norm_mode=norm_mode, max_a=max_a_poly)

        # Decoder
        Act = ACTIVATIONS[act_type]

        self.up4 = nn.ConvTranspose2d(filters[5], filters[4], 2, stride=2)
        self.dec4 = ConvBlock(filters[4] + filters[4], filters[4],
                              norm_type=norm_type, act_type=block_act_type('dec4'), clamp_values=cv('dec4'),
                              norm_mode=norm_mode, max_a=max_a_poly)

        self.up3 = nn.ConvTranspose2d(filters[4], filters[3], 2, stride=2)
        self.dec3 = ConvBlock(filters[3] + filters[3], filters[3],
                              norm_type=norm_type, act_type=block_act_type('dec3'), clamp_values=cv('dec3'),
                              norm_mode=norm_mode, max_a=max_a_poly)

        self.up2 = nn.ConvTranspose2d(filters[3], filters[2], 2, stride=2)
        self.dec2 = ConvBlock(filters[2] + filters[2], filters[2],
                              norm_type=norm_type, act_type=block_act_type('dec2'), clamp_values=cv('dec2'),
                              norm_mode=norm_mode, max_a=max_a_poly)

        self.up1 = nn.ConvTranspose2d(filters[2], filters[1], 2, stride=2)
        self.dec1 = ConvBlock(filters[1] + filters[1], filters[1],
                              norm_type=norm_type, act_type=block_act_type('dec1'), clamp_values=cv('dec1'),
                              norm_mode=norm_mode, max_a=max_a_poly)

        self.up0 = nn.ConvTranspose2d(filters[1], filters[0], 2, stride=2)
        self.dec0 = ConvBlock(filters[0] + filters[0], filters[0],
                              norm_type=norm_type, act_type=block_act_type('dec0'), clamp_values=cv('dec0'),
                              norm_mode=norm_mode, max_a=max_a_poly)

        # Output
        self.out_conv = nn.Conv2d(filters[0], num_classes, 1)

    def forward(self, x):
        # Encoder
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(e5), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        d0 = self.dec0(torch.cat([self.up0(d1), e0], dim=1))

        return self.out_conv(d0)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    for act in ['identity', 'linear', 'squared', 'poly']:
        for norm in ['none', 'batch']:
            model = HEFriendlyUNet(act_type=act, norm_type=norm)
            x = torch.randn(1, 1, 256, 224)
            y = model(x)
            n_params = sum(p.numel() for p in model.parameters())
            print(f'act={act:10s} norm={norm:10s} '
                  f'output={tuple(y.shape)}  params={n_params:,}')

    # Sanity check norm_mode su norm_type='instance'
    m_pop = HEFriendlyUNet(act_type='poly', norm_type='instance', norm_mode='population')
    m_pi  = HEFriendlyUNet(act_type='poly', norm_type='instance', norm_mode='per_instance')
    print('norm_mode=population -> track_running_stats =',
          m_pop.enc0.block[1].track_running_stats)   # atteso: True
    print('norm_mode=per_instance -> track_running_stats =',
          m_pi.enc0.block[1].track_running_stats)     # atteso: False