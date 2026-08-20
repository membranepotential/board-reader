---
name: tunnel
description: Expose the local `make dev` stack to the public internet over a Cloudflare quick tunnel, so a phone off the LAN can open /camera, /snapshot and the live page with a real TLS certificate. Use when the user says "/tunnel", "tunnel the dev deployment", "expose board-reader remotely", "share the camera page", "test from outside the LAN", or asks how to reach the viewer from a phone that is not on the local network.
allowed_tools: ["Bash", "Read"]
---

# /tunnel — reach `make dev` from anywhere

Puts the running viewer behind `https://<random>.trycloudflare.com` with a
browser-trusted certificate. That certificate is the point: `/camera` calls
`getUserMedia`, which browsers refuse outside a secure context, and the tunnel's
real cert means no self-signed interstitial to accept on the phone.

## One port is enough

Tunnel **8080 only** — the plain-HTTP viewer bind. All three pages are
same-origin: every inference endpoint they touch is relayed by the viewer itself
(`packages/viewer/src/board_reader_viewer/proxy.py`), so the browser never needs
to reach `:8000`, and nothing is injected into the HTML that could hardcode a LAN
address.

Do **not** tunnel the `--ssl` bind on 8443. `cloudflared` would have to talk
HTTPS to a self-signed origin, which buys nothing — the tunnel already terminates
TLS at Cloudflare's edge. The 8443 bind exists for LAN use, where there is no
edge to do it.

Never tunnel `:8000`. The inference node has no auth of any kind: `/calibrate`
rebuilds the exemplar bank and `/uploads` serves every photo ever ingested.

## Run it

Prerequisite: `cloudflared` (Arch `extra`, already installed) and a running
`make dev`. No Cloudflare account, no login, no `cert.pem` — quick tunnels are
anonymous.

```bash
# 1. the stack, if it is not already up (separate terminal — it runs in foreground)
make dev

# 2. the tunnel
cloudflared tunnel --url http://localhost:8080
```

`cloudflared` prints the assigned hostname in a box within ~10 s. Scrape it
rather than asking the user to read it back:

```bash
cloudflared tunnel --url http://localhost:8080 > /tmp/cfd.log 2>&1 &
sleep 12
grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' /tmp/cfd.log | head -1
```

Hand the user all three URLs, `/camera` first — it is the one that needs the
tunnel most, since the others work fine over plain LAN HTTP:

```
https://<host>.trycloudflare.com/camera     phone camera -> live position
https://<host>.trycloudflare.com/snapshot   one photo, read once
https://<host>.trycloudflare.com/           live viewer
```

## Verify before handing it over

A tunnel that resolves is not a tunnel that works — check the relay, not just the
page:

```bash
H=<host>.trycloudflare.com
curl -sS -o /dev/null -w '%{http_code}\n' https://$H/healthz          # 200 from the viewer
curl -sS -o /dev/null -w '%{http_code}\n' https://$H/camera           # 200, page loads
curl -sS https://$H/api/uploads?limit=1 | head -c 200; echo           # 200 => relay reaches inference
```

The third is the one that matters. The first two pass even when the inference
node is down, because the viewer serves its own static pages.

## Teardown

```bash
pkill -f "cloudflared tunnel"
sleep 1
pgrep -f "cloudflared tunnel" >/dev/null && pkill -9 -f "cloudflared tunnel"
```

`cloudflared` does not always go down on SIGTERM — verify rather than assume, or
you leave a public URL onto the user's camera running unattended. Same for the
stack itself: `make stop`, then confirm with
`pgrep -f "board-reader-(inference|viewer)"`.

The hostname dies with the process and is never reissued. Every restart hands out
a different one, so treat the URL as single-session and re-scrape it each time.

## What this does not do, and why

**No custom domain.** A quick tunnel cannot serve a hostname of your own, at
any layer. Measured against a live tunnel with a custom domain pointed at it:

| Attempt | Result |
|---|---|
| TLS with SNI for the custom hostname | `handshake failure` — no cert is offered at all, so unlike a self-signed LAN cert there is no interstitial to click through |
| Same with certs ignored (`curl -k`) | identical failure; it breaks before certs are involved |
| Plain HTTP on port 80 | `error code: 1001` |
| Valid quick-tunnel SNI, custom `Host:` header | `403` — the edge routes quick tunnels strictly by their assigned hostname |

Cloudflare's edge only serves hostnames belonging to a zone in a Cloudflare
account, and `<uuid>.cfargotunnel.com` is not publicly resolvable, so a CNAME to
it from a third-party DNS provider resolves to nothing. Attaching a custom
hostname needs a named tunnel plus the zone on Cloudflare; bridging a zone hosted
elsewhere is a paid plan either way (partial/CNAME setup is Business+, delegating
a subdomain as its own zone is Enterprise). **Do not re-propose a custom
hostname for this skill — take the assigned quick-tunnel URL.**

**No access control.** Anyone with the URL reaches the page, and through it the
inference node's relayed API. The URL is unguessable and short-lived, which is
the whole of the protection. Do not leave a tunnel running unattended, and do not
paste the URL anywhere durable.
