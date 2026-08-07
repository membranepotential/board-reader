"""Tests for the viewer's self-signed certificate (``--ssl``).

The certificate exists for exactly one reason — ``getUserMedia`` refuses to work
outside a secure context — so what matters is that the phone can actually reach the
name the operator types, and that it is asked to trust the certificate once rather
than on every restart.
"""

from __future__ import annotations

import datetime as dt
import socket

from cryptography import x509

from board_reader_viewer import tls


def _load(cert: tls.Certificate) -> x509.Certificate:
    return x509.load_pem_x509_certificate(cert.certfile.read_bytes())


def _san_values(certificate: x509.Certificate) -> set[str]:
    san = certificate.extensions.get_extension_for_class(x509.SubjectAlternativeName)
    return {str(entry.value) for entry in san.value}


def test_generates_a_parseable_pair(tmp_path):
    cert = tls.ensure_certificate(tmp_path)
    assert cert.generated
    assert cert.certfile.exists() and cert.keyfile.exists()
    parsed = _load(cert)
    assert parsed.not_valid_after_utc > dt.datetime.now(dt.timezone.utc)
    # Backdated slightly, so a phone whose clock runs behind does not meet a
    # not-yet-valid certificate.
    assert parsed.not_valid_before_utc < dt.datetime.now(dt.timezone.utc)


def test_covers_localhost_loopback_hostname_and_every_lan_address(tmp_path):
    """The operator types whichever address their phone shares a network with; every
    one of them has to be in the certificate or it is rejected before they are ever
    offered the choice to trust it."""
    names = _san_values(_load(tls.ensure_certificate(tmp_path)))
    assert "localhost" in names
    assert "127.0.0.1" in names
    assert socket.gethostname() in names
    for address in tls.local_ipv4_addresses():
        assert address in names
    assert set(tls.subject_names()) <= names


def test_the_key_is_not_world_readable(tmp_path):
    cert = tls.ensure_certificate(tmp_path)
    assert cert.keyfile.stat().st_mode & 0o077 == 0


def test_a_cached_pair_is_reused(tmp_path):
    """One certificate acceptance per machine, not per restart."""
    first = tls.ensure_certificate(tmp_path)
    second = tls.ensure_certificate(tmp_path)
    assert first.generated and not second.generated
    assert second.certfile.read_bytes() == first.certfile.read_bytes()
    assert second.keyfile.read_bytes() == first.keyfile.read_bytes()


def test_a_pair_that_no_longer_covers_this_host_is_replaced(tmp_path, monkeypatch):
    """Moving between networks changes the LAN address. A cached certificate that does
    not name the address now in use is worse than none — the phone refuses it outright
    — so coverage, not just freshness, decides whether the cache can serve."""
    monkeypatch.setattr(tls, "local_ipv4_addresses", lambda: ("10.9.9.9",))
    stale = tls.ensure_certificate(tmp_path)
    assert "10.9.9.9" in _san_values(_load(stale))

    monkeypatch.setattr(tls, "local_ipv4_addresses", lambda: ("192.168.5.5",))
    fresh = tls.ensure_certificate(tmp_path)
    assert fresh.generated
    assert "192.168.5.5" in _san_values(_load(fresh))


def test_a_corrupt_cache_is_replaced_rather_than_fatal(tmp_path):
    tls.ensure_certificate(tmp_path)
    (tmp_path / "cert.pem").write_text("not a certificate")
    assert tls.ensure_certificate(tmp_path).generated


def test_cache_dir_follows_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert tls.cache_dir() == tmp_path / "board-reader" / "viewer-tls"


def test_camera_urls_offer_every_local_address_for_a_wildcard_bind(monkeypatch):
    monkeypatch.setattr(tls, "local_ipv4_addresses", lambda: ("192.168.1.7",))
    urls = tls.camera_urls("0.0.0.0", 8080, scheme="https")
    assert "https://192.168.1.7:8080/camera" in urls
    assert "https://localhost:8080/camera" in urls


def test_camera_urls_do_not_invent_reachability_for_a_specific_bind(monkeypatch):
    """Bound to loopback the server is reachable at loopback and nowhere else; offering
    a LAN URL would send the operator to type an address that cannot answer."""
    monkeypatch.setattr(tls, "local_ipv4_addresses", lambda: ("192.168.1.7",))
    assert tls.camera_urls("127.0.0.1", 8080, scheme="http") == (
        "http://127.0.0.1:8080/camera",
    )
