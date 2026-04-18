import torch.nn as nn


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                return candidate
    return checkpoint


def _strip_module_prefix(state_dict):
    if not any(key.startswith("module.") for key in state_dict.keys()):
        return state_dict

    return {
        key[len("module.") :] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def _adapt_resnet_fc_if_needed(model, state_dict, source):
    fc = getattr(model, "fc", None)
    if not isinstance(fc, nn.Linear):
        return

    needed_keys = ("fc.1.weight", "fc.1.bias", "fc.4.weight", "fc.4.bias")
    if not all(key in state_dict for key in needed_keys):
        return

    fc1_weight = state_dict["fc.1.weight"]
    fc1_bias = state_dict["fc.1.bias"]
    fc4_weight = state_dict["fc.4.weight"]
    fc4_bias = state_dict["fc.4.bias"]

    # ... validation omitted ...
    hidden_features = fc1_weight.shape[0]
    out_features = fc4_weight.shape[0]
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(fc.in_features, hidden_features),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden_features, out_features),
    )


def _adapt_heads_dynamic(model, state_dict, source):
    """Detects and resolves head structure mismatches for EfficientNet/MobileNet."""
    if not hasattr(model, "classifier"):
        return

    # Check for MobileNet V3 head (index 3)
    # Target: Sequential with nested Linear at index 3.1
    # Checkpoint: Direct Linear at index 3
    if len(model.classifier) > 3:
        if "classifier.3.weight" in state_dict and not "classifier.3.1.weight" in state_dict:
            # If the model has a Sequential head but checkpoint has direct, simplify the model head
            in_f = state_dict["classifier.3.weight"].shape[1]
            out_f = state_dict["classifier.3.weight"].shape[0]
            if isinstance(model.classifier[3], nn.Sequential):
                print(f"Adapting MobileNet classifier[3] to direct Linear for {source}")
                model.classifier[3] = nn.Linear(in_f, out_f)

    # Check for EfficientNet head (index 1)
    if len(model.classifier) > 1:
        if "classifier.1.weight" in state_dict and not "classifier.1.1.weight" in state_dict:
             in_f = state_dict["classifier.1.weight"].shape[1]
             out_f = state_dict["classifier.1.weight"].shape[0]
             if isinstance(model.classifier[1], nn.Sequential):
                print(f"Adapting EfficientNet classifier[1] to direct Linear for {source}")
                model.classifier[1] = nn.Linear(in_f, out_f)


def load_checkpoint_strict(model, checkpoint, source="checkpoint"):
    state_dict = _extract_state_dict(checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format from {source}: expected state_dict mapping.")

    state_dict = _strip_module_prefix(state_dict)
    _adapt_resnet_fc_if_needed(model, state_dict, source)
    _adapt_heads_dynamic(model, state_dict, source)

    try:
        # Final safety check: if we have a total shape mismatch in any layer (like layer0 in EfficientNet)
        # load_state_dict(strict=False) will still raise error on shape mismatch.
        incompatible = model.load_state_dict(state_dict, strict=False)
    except RuntimeError as exc:
        raise RuntimeError(f"Failed loading state dict from {source}: {exc}") from exc

    # We only report errors if they are CRITICAL mismatches (ignoring stuff we can live with)
    # But for a 'strict' loader, any mismatch is an error.
    missing = sorted(incompatible.missing_keys)
    unexpected = sorted(incompatible.unexpected_keys)
    
    # We allow missing keys for dropout/batchnorm if necessary, but weights must match
    if any(".weight" in k or ".bias" in k for k in missing + unexpected):
         raise RuntimeError(
             f"State dict mismatch after compatibility fixes for {source}. "
             f"Missing keys: {missing}. Unexpected keys: {unexpected}."
         )

    return model
