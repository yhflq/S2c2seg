import hashlib
import json

CACHE_SCHEMA_VERSION = 1


def _digest(payload):
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def tensor_fingerprint(tensor):
    array = tensor.detach().to("cpu").contiguous()
    header = ("%s|%s" % (array.dtype, tuple(array.shape))).encode()
    return _digest(header + array.numpy().tobytes())


def config_fingerprint(config_signature, extra=None):
    payload = {"schema": CACHE_SCHEMA_VERSION, "config": config_signature}
    if extra:
        payload["extra"] = extra
    encoded = json.dumps(payload, sort_keys=True, default=str).encode()
    return _digest(encoded)


def vocabulary_fingerprint(words):
    return _digest("␟".join(words).encode())


def crop_key(image_fingerprint, config_hash, vocabulary_hash):
    return "%s-%s-%s" % (config_hash, vocabulary_hash, image_fingerprint)
