"""
crypto/test_set_guard.py

La regola del progetto, scritta in evaluate_on_test_set.py, per il TEST SET UFFICIALE ACDC (patient101-150):
 - NON si usa per scegliere tra configurazioni (quello e' compito della validazione, fold 0);
 - va bene per una verifica tecnica una tantum su una configurazione "chiusa", o per la valutazione FINALE a fine progetto;
 - ogni uso va giustificato con una nota e viene registrato nel log, con data, cosi' resta uno storico.
Questo modulo applica la stessa regola agli strumenti nuovi (golden_approx_dice.py, pack_for_zeus.py), che a differenza di
evaluate_on_test_set.py funzionano con qualunque numero di stage, skip sum, norm bypassate e radice inversa approssimata.
"""

import json
import os
import time


def confirm_test_set_use(note, skip_confirmation=False, what=""):
    print("=" * 70)
    print("ATTENZIONE: stai per usare il VERO TEST SET (patient101-150)")
    print("=" * 70)
    print("Regola concordata: questo NON deve servire a scegliere tra configurazioni.")
    print("Va bene solo per una verifica tecnica una tantum su una configurazione CHIUSA,")
    print("o per la valutazione FINALE a fine progetto.")
    if what:
        print(f"\nUso previsto: {what}")
    print(f'Nota fornita: "{note}"\n')
    if skip_confirmation:
        print("(conferma saltata con --skip_confirmation)")
        return True
    ans = input('Confermi di voler procedere? Scrivi "si" per continuare: ')
    if ans.strip().lower() not in ("si", "sì", "yes", "y"):
        print("Annullato.")
        return False
    return True


def append_log(log_file, entry):
    entry = dict(entry)
    entry.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))
    data = []
    if os.path.exists(log_file):
        with open(log_file) as f:
            data = json.load(f)
    data.append(entry)
    with open(log_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  (uso del test set registrato in {log_file})")