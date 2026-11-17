"""Conservative self-reported receipt classification; no host/model attestation."""
import math
import re

HEX256 = re.compile(r"^[0-9a-f]{64}$")
FIELDS = ("taught_exact", "unseen_paraphrase", "untaught_control")

def numeric(value, name):
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError("invalid numeric " + name)
    return float(value)

def classify_self_reported(training, receiver):
    """Never promote nonblind loss changes to verified generalization."""
    if type(training) is not dict or type(receiver) is not dict:
        raise ValueError("receipt objects required")
    for a, b in (("source", "model"), ("revision", "revision")):
        if type(training.get(a)) is not str or training.get(a) != receiver.get(b):
            raise ValueError("model identity/revision mismatch")
    if type(training.get("steps")) is not int or training["steps"] <= 0:
        raise ValueError("missing actual donor updates")
    if receiver.get("training_steps_in_receiver") != 0 or type(receiver.get("training_steps_in_receiver")) is not int:
        raise ValueError("receiver updated during supposed readback")
    if receiver.get("generalization_established") is not False:
        raise ValueError("invalid generalization claim")
    digest = training.get("adapter_files_sha256")
    if type(digest) is not dict:
        raise ValueError("adapter hash map missing")
    expected = digest.get("adapter_model.safetensors")
    if type(expected) is not str or not HEX256.fullmatch(expected):
        raise ValueError("adapter digest absent or malformed")
    if receiver.get("adapter_sha256") != expected:
        raise ValueError("adapter digest mismatch")
    if numeric(training.get("lora_tensor_max_change"), "lora_delta") <= 0:
        raise ValueError("no measured adapter update")
    numeric(training.get("loss_before"), "train_before")
    numeric(training.get("loss_after"), "train_after")
    before, after = receiver.get("base_losses"), receiver.get("adapter_losses")
    if type(before) is not dict or type(after) is not dict:
        raise ValueError("paired receiver losses required")
    a, b = {}, {}
    for key in FIELDS:
        a[key] = numeric(before.get(key), "base." + key)
        b[key] = numeric(after.get(key), "adapter." + key)
    learned = b["taught_exact"] < a["taught_exact"]
    confounded = b["untaught_control"] < a["untaught_control"]
    return {
        "evidence": "SELF_REPORTED_RECEIPT_CONSISTENCY_ONLY",
        "classification": "PERSISTENT_ADAPTER_EFFECT_UNBLINDED" if learned else "NO_PERSISTENT_GAIN_OBSERVED",
        "untaught_target_also_improved": confounded,
        "unseen_paraphrase_target_improved": b["unseen_paraphrase"] < a["unseen_paraphrase"],
        "generalization_proven": False,
        "neurosync_novelty_proven": False,
    }
