# Test-only TLS fixtures

Self-signed certificates and private keys used ONLY by the remote-worker
TLS tests (tests/test_remote.py). They protect nothing: the keys are
public by design and must never be deployed.

- `service-cert.pem` / `service-key.pem`: the daemon's identity; the
  client pins `service-cert.pem` as its CA file.
- `other-cert.pem` / `other-key.pem`: an unrelated identity, used to
  prove that a client pinning the wrong certificate is refused.

Regenerate (100-year validity, SAN covers localhost and 127.0.0.1):

    openssl req -x509 -newkey rsa:2048 -nodes -keyout service-key.pem \
        -out service-cert.pem -days 36500 -subj "/CN=dinkster-test-service" \
        -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"

    openssl req -x509 -newkey rsa:2048 -nodes -keyout other-key.pem \
        -out other-cert.pem -days 36500 -subj "/CN=dinkster-test-other" \
        -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"
