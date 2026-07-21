"""
ravnest.security.tls — TLS/mTLS certificate helpers for Ravnest nodes.

Provides a thin wrapper around the ``cryptography`` library to:
- Generate self-signed CA certificates.
- Issue node-specific leaf certificates signed by the CA.
- Build ``ssl.SSLContext`` objects for both server and client sides
  of an mTLS connection.

Requires: ``pip install cryptography``

If ``cryptography`` is not installed, all functions raise ``ImportError``
with a helpful message.  The rest of Ravnest remains fully usable without it.

Usage
-----
    from ravnest.security.tls import CertBundle, generate_ca, generate_node_cert

    ca   = generate_ca("ravnest-ca")
    node = generate_node_cert(ca, "node-worker-1")

    # Server context (NodeServer / GatewayServer)
    ssl_ctx = node.server_ssl_context(ca_bundle=ca)

    # Client context (NodeClient)
    ssl_ctx = node.client_ssl_context(ca_bundle=ca)

    # Pass to aiohttp:
    site = web.TCPSite(runner, "0.0.0.0", 8765, ssl_context=ssl_ctx)

Mutual TLS
----------
When ``ca_bundle`` is provided to :meth:`CertBundle.server_ssl_context`,
client certificate verification is required (``CERT_REQUIRED``).  This means
every NodeClient must present its own leaf certificate to connect — preventing
unauthenticated nodes from joining the mesh.

Cert storage
------------
Both :class:`CertBundle` cert and key are kept in memory by default.  Call
:meth:`CertBundle.save` to write PEM files to disk for persistence across
restarts.
"""

from __future__ import annotations

import ipaddress
import ssl
import tempfile
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Optional

_MISSING_MSG = (
    "The 'cryptography' package is required for TLS helpers. "
    "Install it with:  pip install cryptography"
)


def _require_cryptography():
    try:
        import cryptography  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        raise ImportError(_MISSING_MSG)


@dataclass
class CertBundle:
    """
    Holds a certificate + private key pair in PEM format.

    Attributes
    ----------
    cert_pem:  PEM-encoded X.509 certificate bytes.
    key_pem:   PEM-encoded private key bytes (unencrypted).
    cn:        Common name of this certificate.
    is_ca:     True if this is a CA certificate.
    """
    cert_pem: bytes
    key_pem:  bytes
    cn:       str
    is_ca:    bool = False

    # ── ssl.SSLContext builders ───────────────────────────────────────────

    def server_ssl_context(
        self,
        ca_bundle: Optional["CertBundle"] = None,
    ) -> ssl.SSLContext:
        """
        Build an ``ssl.SSLContext`` for a TLS server.

        If ``ca_bundle`` is provided, mutual TLS is enabled: clients must
        present a certificate signed by ``ca_bundle``.
        """
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self._load_into(ctx)
        if ca_bundle is not None:
            ca_file = _write_temp_pem(ca_bundle.cert_pem)
            try:
                ctx.load_verify_locations(ca_file)
                ctx.verify_mode = ssl.CERT_REQUIRED
            finally:
                os.unlink(ca_file)
        return ctx

    def client_ssl_context(
        self,
        ca_bundle: Optional["CertBundle"] = None,
        server_hostname: Optional[str]    = None,
    ) -> ssl.SSLContext:
        """
        Build an ``ssl.SSLContext`` for a TLS client.

        If ``ca_bundle`` is provided, the server certificate is verified
        against that CA.  Pass ``server_hostname`` when the CN differs from
        the hostname used to connect.
        """
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        self._load_into(ctx)
        if ca_bundle is not None:
            ca_file = _write_temp_pem(ca_bundle.cert_pem)
            try:
                ctx.load_verify_locations(ca_file)
            finally:
                os.unlink(ca_file)
            ctx.check_hostname = server_hostname is not None
            ctx.verify_mode    = ssl.CERT_REQUIRED
        else:
            ctx.check_hostname = False
            ctx.verify_mode    = ssl.CERT_NONE
        return ctx

    def _load_into(self, ctx: ssl.SSLContext) -> None:
        cert_file = _write_temp_pem(self.cert_pem)
        key_file  = _write_temp_pem(self.key_pem)
        try:
            ctx.load_cert_chain(certfile=cert_file, keyfile=key_file)
        finally:
            os.unlink(cert_file)
            os.unlink(key_file)

    # ── persistence ───────────────────────────────────────────────────────

    def save(self, directory: str, name: Optional[str] = None) -> tuple[str, str]:
        """
        Write cert and key PEM files to ``directory``.

        Returns (cert_path, key_path).
        """
        stem      = name or self.cn.replace(" ", "_")
        cert_path = os.path.join(directory, f"{stem}.crt")
        key_path  = os.path.join(directory, f"{stem}.key")
        with open(cert_path, "wb") as f:
            f.write(self.cert_pem)
        with open(key_path,  "wb") as f:
            f.write(self.key_pem)
        return cert_path, key_path

    @classmethod
    def load(cls, cert_path: str, key_path: str,
             is_ca: bool = False) -> "CertBundle":
        """Load a CertBundle from PEM files on disk."""
        with open(cert_path, "rb") as f:
            cert_pem = f.read()
        with open(key_path, "rb") as f:
            key_pem = f.read()
        # Extract CN from cert
        _require_cryptography()
        from cryptography import x509
        cert = x509.load_pem_x509_certificate(cert_pem)
        cn   = cert.subject.get_attributes_for_oid(
            x509.NameOID.COMMON_NAME
        )[0].value
        return cls(cert_pem=cert_pem, key_pem=key_pem, cn=cn, is_ca=is_ca)


# ── certificate generation ────────────────────────────────────────────────────

def generate_ca(
    cn:       str  = "Ravnest CA",
    days:     int  = 3650,
    key_size: int  = 2048,
) -> CertBundle:
    """
    Generate a self-signed CA certificate.

    Args:
        cn:       Common name (e.g. ``"Ravnest CA"``).
        days:     Validity period in days (default 10 years).
        key_size: RSA key size (default 2048 bits).

    Returns:
        A :class:`CertBundle` where ``is_ca=True``.
    """
    _require_cryptography()
    from cryptography                                import x509
    from cryptography.x509.oid                      import NameOID, ExtendedKeyUsageOID
    from cryptography.hazmat.primitives             import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric  import rsa

    key  = rsa.generate_private_key(public_exponent=65537, key_size=key_size)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])
    now  = datetime.now(timezone.utc)

    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + timedelta(days=days))
        .add_extension(
            x509.BasicConstraints(ca=True, path_length=None), critical=True
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(key.public_key()),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )

    return CertBundle(
        cert_pem = cert.public_bytes(serialization.Encoding.PEM),
        key_pem  = key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        ),
        cn    = cn,
        is_ca = True,
    )


def generate_node_cert(
    ca_bundle:   CertBundle,
    node_id:     str,
    days:        int           = 365,
    key_size:    int           = 2048,
    san_ips:     Optional[List[str]] = None,
    san_dns:     Optional[List[str]] = None,
) -> CertBundle:
    """
    Issue a leaf certificate for a Ravnest node, signed by ``ca_bundle``.

    Args:
        ca_bundle:  The CA certificate + key to sign with.
        node_id:    Identifier used as the certificate CN (e.g. ``"node-worker-1"``).
        days:       Validity period (default 1 year).
        key_size:   RSA key size.
        san_ips:    Additional IP SANs (e.g. ``["192.168.1.10", "127.0.0.1"]``).
        san_dns:    Additional DNS SANs (e.g. ``["worker1.internal"]``).

    Returns:
        A :class:`CertBundle` for the node.
    """
    _require_cryptography()
    from cryptography                                import x509
    from cryptography.x509.oid                      import NameOID
    from cryptography.hazmat.primitives             import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric  import rsa

    ca_cert = x509.load_pem_x509_certificate(ca_bundle.cert_pem)
    ca_key  = serialization.load_pem_private_key(ca_bundle.key_pem, password=None)
    key     = rsa.generate_private_key(public_exponent=65537, key_size=key_size)
    name    = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, node_id)])
    now     = datetime.now(timezone.utc)

    # Subject Alternative Names
    san_list: List[x509.GeneralName] = [x509.DNSName(node_id)]
    for ip in (san_ips or []):
        san_list.append(x509.IPAddress(ipaddress.ip_address(ip)))
    for dns in (san_dns or []):
        san_list.append(x509.DNSName(dns))

    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(ca_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + timedelta(days=days))
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None), critical=True
        )
        .add_extension(
            x509.SubjectAlternativeName(san_list), critical=False
        )
        .sign(ca_key, hashes.SHA256())
    )

    return CertBundle(
        cert_pem = cert.public_bytes(serialization.Encoding.PEM),
        key_pem  = key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        ),
        cn    = node_id,
        is_ca = False,
    )


def generate_self_signed(
    cn:       str  = "localhost",
    days:     int  = 365,
    key_size: int  = 2048,
    san_ips:  Optional[List[str]] = None,
    san_dns:  Optional[List[str]] = None,
) -> CertBundle:
    """
    Shortcut: generate a self-signed cert (CA + leaf in one step).

    Suitable for development and testing when you don't need a separate CA.
    """
    ca   = generate_ca(cn=f"{cn}-ca", days=days, key_size=key_size)
    node = generate_node_cert(ca, cn, days=days, key_size=key_size,
                              san_ips=san_ips, san_dns=san_dns)
    return node


# ── helpers ───────────────────────────────────────────────────────────────────

def _write_temp_pem(pem: bytes) -> str:
    """Write PEM bytes to a temp file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".pem")
    try:
        os.write(fd, pem)
    finally:
        os.close(fd)
    return path
