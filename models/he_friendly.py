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

    GATING MORBIDO SU 'a' (soft_max_a, richiesta di Aurora dopo la scoperta
    dello squilibrio nei gradienti skip/upsampling): a differenza di
    max_a (vincolo strutturale rigido, sempre attivo) e di act_overrides
    (commutazione netta e permanente a grado 1 su interi blocchi, decisa
    a priori), qui l'idea e' lasciare che ogni singola PolyAct si comporti
    normalmente (quadratica piena) quando il suo 'a' resta in un range
    sicuro, e "spegnersi" GRADUALMENTE verso un comportamento lineare
    (bx+c) SOLO se 'a' cresce oltre una soglia -- durante il training,
    layer per layer, senza deciderlo a priori su quali blocchi.

    Implementato come un fattore di gate moltiplicato sul termine
    quadratico: gate = sigmoid(soft_sharpness * (soft_max_a - |a|)).
    Per |a| << soft_max_a, gate tende a 1 (quadratico pieno, comportamento
    normale). Per |a| >> soft_max_a, gate tende a 0 (il termine ax^2
    sparisce, resta bx+c). La transizione e' liscia e derivabile ovunque
    (a differenza di un if/else netto, che introdurrebbe una discontinuita'
    nel gradiente proprio nel punto critico in cui interviene) -- il
    training non subisce salti improvvisi di comportamento.

    soft_sharpness controlla quanto e' brusca la transizione: valori alti
    (es. 20-50) la rendono quasi un interruttore netto ma ancora derivabile;
    valori bassi (es. 2-5) la rendono molto graduale. Se soft_max_a e' None
    (default), questo meccanismo e' completamente disattivato -- nessun
    effetto sul comportamento originale, indipendentemente da max_a.

    NOTA su HE: questo gating agisce SOLO sul valore di 'a' usato nel
    forward durante il training. A fine training, ogni PolyAct ha
    semplicemente i suoi coefficienti a,b,c fissi appresi (magari alcuni
    con 'a' effettivamente vicino a zero se il gate si e' spento durante
    l'apprendimento) -- l'inferenza HE resta identica a prima (stesso
    identico polinomio ax^2+bx+c per layer, nessuna operazione aggiuntiva
    da implementare in CKKS). Il gating e' un meccanismo di TRAINING, non
    di inferenza.
    """
    def __init__(self, clamp_value: float = 50.0, max_a: float = None,
                 soft_max_a: float = None, soft_sharpness: float = 10.0):
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
        self.soft_max_a = soft_max_a
        self.soft_sharpness = soft_sharpness

    def current_a(self):
        """Valore effettivo di 'a' da usare nel forward e nella diagnostica,
        sia in modalita' libera che vincolata. NON include il gating morbido
        (vedi effective_a_for_forward()) -- questo resta il valore "grezzo"
        del coefficiente appreso, utile per diagnostica/logging inalterati."""
        if self.max_a is None:
            return self.a
        return self.max_a * torch.tanh(self.raw_a)

    def gate_value(self):
        """Il fattore di gate corrente (tra 0 e 1) applicato al termine
        quadratico. Ritorna 1.0 (nessun effetto) se soft_max_a e' None.
        Esposto separatamente dalla diagnostica, cosi' si puo' monitorare
        QUANDO e DOVE il gate comincia a intervenire durante il training."""
        if self.soft_max_a is None:
            return torch.tensor(1.0)
        a = self.current_a()
        return torch.sigmoid(self.soft_sharpness * (self.soft_max_a - a.abs()))

    def forward(self, x):
        a = self.current_a()
        gate = self.gate_value()
        self.last_gate = gate.detach() if torch.is_tensor(gate) else gate
        out = (gate * a) * x * x + self.b * x + self.c
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
# Weight Standardization (Qiao et al., 2019) -- opzionale
# ---------------------------------------------------------------------------

class WSConv2d(nn.Conv2d):
    """
    Conv2d con Weight Standardization: prima di ogni forward, i pesi di
    ciascun canale di output vengono standardizzati (media 0, deviazione
    standard 1) sulle dimensioni (in_channels, kH, kW), poi riscalati da
    un gain per-canale strutturalmente vincolato a restare >= gain_floor
    (vedi sotto) -- una trasformazione sui PARAMETRI, non sulle attivazioni
    (a differenza di BatchNorm/InstanceNorm).

    gain_floor: valore minimo strutturale per 'gain' (vedi forward).
    Impedisce al gain di collassare verso zero durante il training --
    motivato dalla scoperta empirica (crypto/check_ws_sum_norm_collapse.py)
    che un gain libero puo' scivolare fino a ~2e-5 su singoli canali,
    producendo un'uscita quasi costante che fa collassare la varianza
    per-istanza della InstanceNorm successiva (0 -> 1/sqrt(var) esplode).

    IMPORTANTE per la compatibilita' con i checkpoint: un checkpoint
    allenato PRIMA di questo fix ha una chiave 'gain' (gain libero, senza
    floor) invece di 'raw_gain' -- caricarlo qui con strict=False scarta
    silenziosamente quella chiave e lascia raw_gain a zero per ogni
    canale, falsando qualunque valutazione. Per quei checkpoint "storici"
    serve ricostruire la vecchia WSConv2d (vedi crypto/ablate_decoder_
    stages.py, classe LegacyWSConv2d) invece di questa.
    """
    def __init__(self, *args, gain_floor: float = 0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.gain_floor = gain_floor
        # raw_gain e' il parametro libero; gain effettivo = gain_floor + softplus(raw_gain)
        # softplus è sempre >= 0, quindi gain >= gain_floor SEMPRE, qualunque
        # gradiente proponga l'ottimizzatore -- stesso principio di max_a/tanh.
        self.raw_gain = nn.Parameter(torch.zeros(self.out_channels))

    def effective_gain(self):
        return self.gain_floor + nn.functional.softplus(self.raw_gain)

    def forward(self, x):
        weight = self.weight
        out_ch = weight.shape[0]
        w_flat = weight.reshape(out_ch, -1)
        mean = w_flat.mean(dim=1, keepdim=True)
        std = w_flat.std(dim=1, keepdim=True, unbiased=False)
        w_standardized = (w_flat - mean) / (std + 1e-5)
        w_standardized = w_standardized * self.effective_gain().view(-1, 1)
        w_standardized = w_standardized.reshape(weight.shape)
        return nn.functional.conv2d(
            x, w_standardized, self.bias, self.stride,
            self.padding, self.dilation, self.groups
        )


def calibrate_ws_gain(model):
    """Calibra raw_gain tramite la softplus INVERSA, cosi' il gain effettivo
    iniziale coincide con lo std originale dei pesi caricati (evita lo shock
    iniziale), ma 'gain' non potra' mai scendere sotto gain_floor."""
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, WSConv2d):
                w_flat = m.weight.reshape(m.out_channels, -1)
                target_gain = w_flat.std(dim=1, unbiased=False)
                # softplus_inv(y) = log(exp(y) - 1), con clamp per stabilita' numerica
                # quando target_gain e' vicino o sotto gain_floor.
                delta = torch.clamp(target_gain - m.gain_floor, min=1e-6)
                m.raw_gain.copy_(torch.log(torch.expm1(delta)))


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

    soft_max_a, soft_sharpness: propagati identici a entrambe le PolyAct
    del blocco -- vedi PolyAct per la spiegazione del gating morbido.

    weight_standardization: se True, entrambe le Conv2d del blocco usano
    WSConv2d invece di nn.Conv2d -- vedi WSConv2d per la motivazione.
    Default False (invariato, comportamento originale).
    """
    def __init__(self, in_ch, out_ch, stride=1,
                 norm_type='none', act_type='poly', clamp_values=None,
                 norm_mode='population', max_a=None,
                 soft_max_a=None, soft_sharpness=10.0,
                 weight_standardization=False):
        super().__init__()
        Act = ACTIVATIONS[act_type]
        Conv = WSConv2d if weight_standardization else nn.Conv2d

        def make_act():
            if act_type == 'poly':
                kwargs = {'max_a': max_a, 'soft_max_a': soft_max_a,
                         'soft_sharpness': soft_sharpness}
                if clamp_values is not None:
                    idx = make_act.counter
                    make_act.counter += 1
                    kwargs['clamp_value'] = clamp_values[idx]
                return Act(**kwargs)
            return Act()
        make_act.counter = 0

        self.block = nn.Sequential(
            Conv(in_ch, out_ch, 3, stride=stride, padding=1, bias=True),
            get_norm(norm_type, out_ch, norm_mode),
            make_act(),
            Conv(out_ch, out_ch, 3, stride=1, padding=1, bias=True),
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
        skip_mode:    'concat' | 'sum' (vedi sotto)
        filters:      lista di 6 interi, canali per stage encoder (vedi sotto)
    """

    def __init__(self, in_channels=1, num_classes=4,
                 act_type='poly', norm_type='none', clamp_values=None,
                 norm_mode='population', max_a_poly=None, act_overrides=None,
                 soft_max_a=None, soft_sharpness=10.0, skip_mode='concat',
                 weight_standardization=False, filters=None):
        """
        weight_standardization: se True (default False, invariato), ogni
        Conv2d "normale" della rete (dentro i ConvBlock di encoder/decoder,
        e il layer di output finale) usa WSConv2d invece di nn.Conv2d --
        vedi WSConv2d per la motivazione completa. Le ConvTranspose2d di
        upsampling (up4..up0) NON sono affette (scelta di scope, vedi
        WSConv2d). A differenza di skip_mode, questo NON cambia la forma
        dei pesi -- un checkpoint allenato senza WS resta caricabile in un
        modello con WS attivata (e viceversa) senza errori di shape, ma il
        comportamento numerico cambia, quindi le prestazioni vanno
        riverificate.

        filters: lista di 6 interi [enc0, enc1, enc2, enc3, enc4, enc5] --
        numero di canali per stage encoder (dec4..dec0 usano enc4..enc0 in
        ordine inverso, essendo un'architettura simmetrica). Default None:
        usa [32, 64, 128, 256, 512, 512], il dimensionamento originale.

        Motivazione (crypto/ablate_decoder_width.py, richiesta di Aurora sul
        dimensionamento): l'ablazione per canale ha mostrato ridondanza molto
        alta negli stage a bassa risoluzione spaziale (enc4, enc5, dec4 --
        tenuta quasi intatta anche a 1/8 della larghezza), e ridondanza bassa
        negli stage ad alta risoluzione (enc0, dec0, dec1, dec2 -- crollo gia'
        a 1/2 o 1/4). Questo parametro permette di testare configurazioni piu'
        strette in modo mirato, invece di ridurre l'intera rete uniformemente.
        ATTENZIONE: cambia le shape dei pesi -- richiede training da zero,
        non caricabile via --init_from/--pretrained da un checkpoint con
        filters diversi.

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

        soft_max_a, soft_sharpness: propagati a ogni PolyAct della rete --
        vedi PolyAct per la spiegazione completa del gating morbido
        (richiesta di Aurora: lasciare che ogni layer si comporti in modo
        quadratico pieno quando 'a' resta in un range sicuro, e "spegnersi"
        gradualmente verso un comportamento lineare SOLO se 'a' cresce
        troppo, invece di decidere a priori quali blocchi rendere lineari
        come fa act_overrides). Default None: nessun effetto, comportamento
        originale invariato.

        skip_mode: 'concat' (default, invariato -- comportamento originale:
        le skip connection vengono concatenate lungo i canali, che vengono
        poi raddoppiati prima di ogni ConvBlock del decoder) oppure 'sum'
        (le skip connection vengono SOMMATE invece di concatenate).

        Motivazione per 'sum' (scoperta con crypto/analyze_decoder_
        gradients.py): con la concatenazione, il gradiente che arriva al
        punto di unione dal ramo di upsampling e quello che arriva dalla
        skip connection possono avere magnitudini molto diverse -- misurato
        fino a un rapporto di ~14:1 nell'ultimo stage del decoder (dec0),
        crescente progressivamente lungo tutto il decoder. Con la somma,
        per costruzione matematica (derivata di una somma), i due
        contributi ricevono ESATTAMENTE lo stesso gradiente nel punto di
        unione -- elimina lo squilibrio strutturalmente, non solo lo
        attenua. Bonus: dato che nell'architettura attuale il numero di
        canali dell'upsampling e della skip coincidono gia' esattamente ad
        ogni stage (es. up4 produce filters[4] canali, e4 ne ha altrettanti
        -- verificato, nessuna conv 1x1 di adattamento necessaria), la
        somma NON raddoppia i canali come fa la concatenazione: ogni
        ConvBlock del decoder riceve meta' dei canali in ingresso rispetto
        a 'concat' (es. filters[4] invece di filters[4]+filters[4] per
        dec4) -- meno parametri, e in prospettiva HE meno ciphertext da
        gestire ad ogni stage del decoder.

        ATTENZIONE: 'sum' cambia il numero di canali in ingresso ai
        ConvBlock del decoder -- un checkpoint allenato con skip_mode=
        'concat' NON e' compatibile con un modello costruito con skip_
        mode='sum' (le shape dei pesi della prima Conv2d di ogni blocco
        decoder non coincidono). Va riallenato da zero, non caricato via
        --init_from/--pretrained da un checkpoint 'concat' esistente.
        """
        super().__init__()

        self.act_type  = act_type
        self.norm_type = norm_type
        self.norm_mode = norm_mode
        self.max_a_poly = max_a_poly
        self.act_overrides = act_overrides or {}
        self.soft_max_a = soft_max_a
        self.soft_sharpness = soft_sharpness
        if skip_mode not in ('concat', 'sum'):
            raise ValueError(f"skip_mode deve essere 'concat' o 'sum', ricevuto: {skip_mode}")
        self.skip_mode = skip_mode
        self.weight_standardization = weight_standardization
        self.filters = filters or [32, 64, 128, 256, 512, 512]

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

        filters = self.filters

        # Encoder
        self.enc0 = ConvBlock(in_channels, filters[0], stride=1,
                              norm_type=norm_type, act_type=block_act_type('enc0'), clamp_values=cv('enc0'),
                              norm_mode=norm_mode, max_a=max_a_poly,
                              soft_max_a=soft_max_a, soft_sharpness=soft_sharpness,
                              weight_standardization=weight_standardization)
        self.enc1 = ConvBlock(filters[0], filters[1], stride=2,
                              norm_type=norm_type, act_type=block_act_type('enc1'), clamp_values=cv('enc1'),
                              norm_mode=norm_mode, max_a=max_a_poly,
                              soft_max_a=soft_max_a, soft_sharpness=soft_sharpness,
                              weight_standardization=weight_standardization)
        self.enc2 = ConvBlock(filters[1], filters[2], stride=2,
                              norm_type=norm_type, act_type=block_act_type('enc2'), clamp_values=cv('enc2'),
                              norm_mode=norm_mode, max_a=max_a_poly,
                              soft_max_a=soft_max_a, soft_sharpness=soft_sharpness,
                              weight_standardization=weight_standardization)
        self.enc3 = ConvBlock(filters[2], filters[3], stride=2,
                              norm_type=norm_type, act_type=block_act_type('enc3'), clamp_values=cv('enc3'),
                              norm_mode=norm_mode, max_a=max_a_poly,
                              soft_max_a=soft_max_a, soft_sharpness=soft_sharpness,
                              weight_standardization=weight_standardization)
        self.enc4 = ConvBlock(filters[3], filters[4], stride=2,
                              norm_type=norm_type, act_type=block_act_type('enc4'), clamp_values=cv('enc4'),
                              norm_mode=norm_mode, max_a=max_a_poly,
                              soft_max_a=soft_max_a, soft_sharpness=soft_sharpness,
                              weight_standardization=weight_standardization)
        self.enc5 = ConvBlock(filters[4], filters[5], stride=2,
                              norm_type=norm_type, act_type=block_act_type('enc5'), clamp_values=cv('enc5'),
                              norm_mode=norm_mode, max_a=max_a_poly,
                              soft_max_a=soft_max_a, soft_sharpness=soft_sharpness,
                              weight_standardization=weight_standardization)

        # Decoder
        Act = ACTIVATIONS[act_type]

        # Con skip_mode='sum', il numero di canali in ingresso a ogni
        # ConvBlock del decoder e' filters[i] (non filters[i]+filters[i])
        # perche' upsampling e skip vengono sommati, non concatenati -- i
        # loro canali coincidono gia' (verificato: up4 produce filters[4],
        # e4 ne ha altrettanti; stessa cosa per ogni altro stage).
        dec_in_mult = 1 if skip_mode == 'sum' else 2

        self.up4 = nn.ConvTranspose2d(filters[5], filters[4], 2, stride=2)
        self.dec4 = ConvBlock(filters[4] * dec_in_mult, filters[4],
                              norm_type=norm_type, act_type=block_act_type('dec4'), clamp_values=cv('dec4'),
                              norm_mode=norm_mode, max_a=max_a_poly,
                              soft_max_a=soft_max_a, soft_sharpness=soft_sharpness,
                              weight_standardization=weight_standardization)

        self.up3 = nn.ConvTranspose2d(filters[4], filters[3], 2, stride=2)
        self.dec3 = ConvBlock(filters[3] * dec_in_mult, filters[3],
                              norm_type=norm_type, act_type=block_act_type('dec3'), clamp_values=cv('dec3'),
                              norm_mode=norm_mode, max_a=max_a_poly,
                              soft_max_a=soft_max_a, soft_sharpness=soft_sharpness,
                              weight_standardization=weight_standardization)

        self.up2 = nn.ConvTranspose2d(filters[3], filters[2], 2, stride=2)
        self.dec2 = ConvBlock(filters[2] * dec_in_mult, filters[2],
                              norm_type=norm_type, act_type=block_act_type('dec2'), clamp_values=cv('dec2'),
                              norm_mode=norm_mode, max_a=max_a_poly,
                              soft_max_a=soft_max_a, soft_sharpness=soft_sharpness,
                              weight_standardization=weight_standardization)

        self.up1 = nn.ConvTranspose2d(filters[2], filters[1], 2, stride=2)
        self.dec1 = ConvBlock(filters[1] * dec_in_mult, filters[1],
                              norm_type=norm_type, act_type=block_act_type('dec1'), clamp_values=cv('dec1'),
                              norm_mode=norm_mode, max_a=max_a_poly,
                              soft_max_a=soft_max_a, soft_sharpness=soft_sharpness,
                              weight_standardization=weight_standardization)

        self.up0 = nn.ConvTranspose2d(filters[1], filters[0], 2, stride=2)
        self.dec0 = ConvBlock(filters[0] * dec_in_mult, filters[0],
                              norm_type=norm_type, act_type=block_act_type('dec0'), clamp_values=cv('dec0'),
                              norm_mode=norm_mode, max_a=max_a_poly,
                              soft_max_a=soft_max_a, soft_sharpness=soft_sharpness,
                              weight_standardization=weight_standardization)

        # Output
        OutConv = WSConv2d if weight_standardization else nn.Conv2d
        self.out_conv = OutConv(filters[0], num_classes, 1)

    def _combine_skip(self, upsampled, skip):
        """Unisce il ramo di upsampling con la skip connection, secondo
        skip_mode -- vedi docstring della classe per la motivazione."""
        if self.skip_mode == 'sum':
            return upsampled + skip
        return torch.cat([upsampled, skip], dim=1)

    def forward(self, x):
        # Encoder
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        # bypass: set di nomi stage ('dec0'..'dec4') il cui calcolo va saltato,
        # sostituendolo con la sola skip connection (l'output dello stage
        # encoder corrispondente, prima dell'upsampling). Default vuoto --
        # nessun effetto sul comportamento esistente, usato solo per
        # l'ablation study (crypto/ablate_decoder_stages.py, richiesta di
        # Aurora dopo la scoperta dello squilibrio nei gradienti skip/
        # upsampling fino a 14:1 in dec0).
        bypass = getattr(self, 'decoder_bypass', set())

        # Decoder with skip connections
        d4 = e4 if 'dec4' in bypass else self.dec4(self._combine_skip(self.up4(e5), e4))
        d3 = e3 if 'dec3' in bypass else self.dec3(self._combine_skip(self.up3(d4), e3))
        d2 = e2 if 'dec2' in bypass else self.dec2(self._combine_skip(self.up2(d3), e2))
        d1 = e1 if 'dec1' in bypass else self.dec1(self._combine_skip(self.up1(d2), e1))
        d0 = e0 if 'dec0' in bypass else self.dec0(self._combine_skip(self.up0(d1), e0))

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

    # Sanity check skip_mode='sum': meno parametri di 'concat' (canali dimezzati nel decoder)
    m_concat = HEFriendlyUNet(act_type='poly', norm_type='instance', skip_mode='concat')
    m_sum = HEFriendlyUNet(act_type='poly', norm_type='instance', skip_mode='sum')
    p_concat = sum(p.numel() for p in m_concat.parameters())
    p_sum = sum(p.numel() for p in m_sum.parameters())
    print(f'skip_mode=concat -> params={p_concat:,}')
    print(f'skip_mode=sum    -> params={p_sum:,}  (atteso: meno di concat)')
    x = torch.randn(1, 1, 256, 224)
    y_sum = m_sum(x)
    print(f'skip_mode=sum forward OK, output shape={tuple(y_sum.shape)}')

    # Sanity check soft_max_a: il gate deve essere vicino a 1 per 'a' piccolo,
    # vicino a 0 per 'a' molto oltre la soglia
    m_soft = HEFriendlyUNet(act_type='poly', norm_type='instance',
                            soft_max_a=0.2, soft_sharpness=30.0)
    x = torch.randn(1, 1, 256, 224)
    y_soft = m_soft(x)
    example_act = m_soft.enc0.block[2]
    print(f'soft_max_a=0.2, a iniziale=0.1 -> gate={example_act.gate_value().item():.4f} '
          f'(atteso: vicino a 1, a e\' sotto la soglia)')
    with torch.no_grad():
        example_act.a.fill_(0.5)  # ben oltre la soglia 0.2
    print(f'soft_max_a=0.2, a forzato a 0.5 -> gate={example_act.gate_value().item():.4f} '
          f'(atteso: vicino a 0, a e\' ben oltre la soglia)')

    # Sanity check weight_standardization: forward funzionante, stessa forma
    # dei pesi di un modello senza WS (compatibilita' di caricamento), ma
    # output numericamente diverso (i pesi vengono standardizzati al volo)
    m_no_ws = HEFriendlyUNet(act_type='poly', norm_type='instance', weight_standardization=False)
    m_ws = HEFriendlyUNet(act_type='poly', norm_type='instance', weight_standardization=True)
    same_shape = m_no_ws.enc0.block[0].weight.shape == m_ws.enc0.block[0].weight.shape
    print(f'\nweight_standardization: stessa forma dei pesi (enc0 conv1) = {same_shape} '
          f'(atteso: True -- compatibilita\' di caricamento checkpoint)')
    x = torch.randn(1, 1, 256, 224)
    y_ws = m_ws(x)
    print(f'weight_standardization=True forward OK, output shape={tuple(y_ws.shape)}')
    # Verifica diretta che WSConv2d standardizzi davvero: media~0, std~1 per canale
    with torch.no_grad():
        conv = m_ws.enc0.block[0]
        w_flat = conv.weight.reshape(conv.weight.shape[0], -1)
        mean = w_flat.mean(dim=1)
        std = w_flat.std(dim=1, unbiased=False)
        w_standardized = (w_flat - mean.unsqueeze(1)) / (std.unsqueeze(1) + 1e-5)
    print(f'WSConv2d: media pesi standardizzati per canale (primi 3) = '
          f'{w_standardized.mean(dim=1)[:3].tolist()} (atteso: vicino a 0)')
    print(f'WSConv2d: std pesi standardizzati per canale (primi 3) = '
          f'{w_standardized.std(dim=1, unbiased=False)[:3].tolist()} (atteso: vicino a 1)')

    # Sanity check filters: rete piu' stretta ha meno parametri, forward OK
    m_default = HEFriendlyUNet(act_type='poly', norm_type='instance')
    m_narrow = HEFriendlyUNet(act_type='poly', norm_type='instance',
                              filters=[32, 64, 128, 256, 128, 64])
    p_default = sum(p.numel() for p in m_default.parameters())
    p_narrow = sum(p.numel() for p in m_narrow.parameters())
    print(f'\nfilters default -> params={p_default:,}')
    print(f'filters ristretti [32,64,128,256,128,64] -> params={p_narrow:,} '
          f'(atteso: meno del default)')
    x = torch.randn(1, 1, 256, 224)
    y_narrow = m_narrow(x)
    print(f'filters ristretti forward OK, output shape={tuple(y_narrow.shape)}')