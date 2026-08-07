"""Self-signed TLS for the viewer, so a phone can use its camera (``--ssl``).

``getUserMedia`` is gated on a *secure context*: over plain http a phone on the LAN
will not even offer the camera, whatever the permissions say. The only ways in are a
real certificate (there is no public hostname here), a port forward
(``adb reverse``, USB-only), or a certificate the operator accepts once. This module
is the third: a self-signed cert covering every address the box currently answers on,
cached so the phone's "accept this certificate" is a one-time cost per machine.

The cached pair is reused until it is nearly expired **or** its SANs no longer cover
the machine's current addresses — moving between networks changes the LAN IP, and a
certificate that does not name the address you typed is rejected before the operator
is ever offered the choice to trust it.
"""

from __future__ import annotations

import datetime as dt
import ipaddress
import os
import socket
from dataclasses import dataclass
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

_VALID_DAYS = 365
# Regenerate before expiry rather than at it, so a long-running box never wakes up
# to a certificate the phone refuses.
_RENEW_AFTER_DAYS = 350


@dataclass(frozen=True)
class Certificate:
    """A cert/key pair on disk, plus what it covers and whether it was just made."""

    certfile: Path
    keyfile: Path
    names: tuple[str, ...]
    generated: bool


def cache_dir() -> Path:
    """Where the pair lives — ``$XDG_CACHE_HOME/board-reader/viewer-tls``."""
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "board-reader" / "viewer-tls"


def local_ipv4_addresses() -> tuple[str, ...]:
    """Every non-loopback IPv4 address this host currently answers on.

    Two sources, unioned, because neither is complete on its own: the address the
    kernel would route a packet out of (the LAN address that matters, found without
    sending anything — a UDP ``connect`` only sets the socket's peer), and whatever
    the hostname resolves to (which catches a second interface, and nothing at all on
    a box whose hostname is only in ``/etc/hosts`` as 127.0.0.1).
    """
    found: list[str] = []
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect(("1.1.1.1", 80))
            found.append(probe.getsockname()[0])
        except OSError:
            pass  # no route out at all: loopback-only box, or no network
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.append(info[4][0])
    except socket.gaierror:
        pass
    ordered = dict.fromkeys(a for a in found if not a.startswith("127."))
    return tuple(ordered)


def subject_names() -> tuple[str, ...]:
    """The names the certificate must cover, in the order they are offered."""
    names = ["localhost", "127.0.0.1", socket.gethostname(), *local_ipv4_addresses()]
    return tuple(dict.fromkeys(n for n in names if n))


def _as_general_name(name: str) -> x509.GeneralName:
    try:
        return x509.IPAddress(ipaddress.ip_address(name))
    except ValueError:
        return x509.DNSName(name)


def _covers(cert: x509.Certificate, names: tuple[str, ...]) -> bool:
    try:
        san = cert.extensions.get_extension_for_class(
            x509.SubjectAlternativeName
        ).value
        eku = cert.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
    except x509.ExtensionNotFound:
        return False  # also retires cached pre-EKU certs, which iOS rejects outright
    if x509.oid.ExtendedKeyUsageOID.SERVER_AUTH not in eku:
        return False
    present = {str(entry.value) for entry in san}
    return set(names) <= present


def _write(directory: Path, names: tuple[str, ...]) -> tuple[Path, Path]:
    """Mint a fresh self-signed pair covering ``names``."""
    # P-256: universally supported by anything with a camera API, and generated in
    # microseconds — an RSA key would make first startup visibly slower for nothing.
    key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, names[0]),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "board-reader"),
        ]
    )
    now = dt.datetime.now(dt.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        # Backdated an hour: a phone whose clock runs slightly behind must not meet a
        # certificate that is not valid yet.
        .not_valid_before(now - dt.timedelta(hours=1))
        .not_valid_after(now + dt.timedelta(days=_VALID_DAYS))
        .add_extension(
            x509.SubjectAlternativeName([_as_general_name(n) for n in names]),
            critical=False,
        )
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        # iOS 13+ requires an explicit serverAuth EKU on TLS server certs and the
        # failure is not user-overridable — without this an iPhone never even
        # offers the "trust this certificate" choice the whole scheme rests on.
        .add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_encipherment=True,
                content_commitment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(key, hashes.SHA256())
    )

    directory.mkdir(parents=True, exist_ok=True)
    certfile, keyfile = directory / "cert.pem", directory / "key.pem"
    certfile.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    # The key is written before its mode is tightened, so create it empty-and-private
    # first rather than leaving a world-readable window.
    keyfile.touch(mode=0o600, exist_ok=True)
    keyfile.chmod(0o600)
    keyfile.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    return certfile, keyfile


def ensure_certificate(directory: Path | None = None) -> Certificate:
    """Return a usable self-signed pair, minting one only when the cache cannot serve.

    Reuse requires all three: both files present, comfortably short of expiry, and
    SANs covering every name this host currently answers on.
    """
    directory = directory if directory is not None else cache_dir()
    names = subject_names()
    certfile, keyfile = directory / "cert.pem", directory / "key.pem"

    if certfile.exists() and keyfile.exists():
        try:
            cached = x509.load_pem_x509_certificate(certfile.read_bytes())
        except ValueError:
            cached = None
        if cached is not None:
            expires = cached.not_valid_after_utc
            fresh = expires - dt.datetime.now(dt.timezone.utc) > dt.timedelta(
                days=_VALID_DAYS - _RENEW_AFTER_DAYS
            )
            if fresh and _covers(cached, names):
                return Certificate(certfile, keyfile, names, generated=False)

    certfile, keyfile = _write(directory, names)
    return Certificate(certfile, keyfile, names, generated=True)


def camera_urls(host: str, port: int, *, scheme: str) -> tuple[str, ...]:
    """The ``/camera`` URLs to type on the phone, given how the server is bound.

    A wildcard bind is reachable at every local address, so every one is offered — the
    operator picks whichever their phone shares a network with. A specific bind is
    reachable at exactly that address and it would be a lie to suggest otherwise.
    """
    if host in ("0.0.0.0", "::", ""):
        hosts = (*local_ipv4_addresses(), "localhost")
    else:
        hosts = (host,)
    return tuple(f"{scheme}://{h}:{port}/camera" for h in hosts)
