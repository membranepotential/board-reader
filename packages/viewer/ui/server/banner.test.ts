import { describe, expect, it } from 'vitest';

import { banner } from './banner.ts';

const node = 'http://127.0.0.1:8000';
const tls = { port: 8443, certfile: '/c/cert.pem', names: ['localhost', '192.168.1.7'], generated: false };

describe('the startup banner', () => {
  it('hands the operator every https camera URL for a wildcard bind, and the cert', () => {
    const text = banner({ host: '0.0.0.0', port: 8080, tls, lan: ['192.168.1.7'], node }).join('\n');
    expect(text).toContain('https://192.168.1.7:8443/camera');
    expect(text).toContain('https://localhost:8443/camera');
    expect(text).toContain('accept the warning once');
    expect(text).toContain('reusing /c/cert.pem');
    expect(text).toContain('valid for localhost, 192.168.1.7');
    // The phone has no controls; the operator configures it from this machine.
    expect(text).toContain('http://localhost:8080/control');
    expect(text).toContain('http://192.168.1.7:8080/snapshot');
    expect(text).toContain('http://192.168.1.7:8080/');
    expect(text).toContain('http://127.0.0.1:8000');
  });

  it('says why the camera will refuse without a TLS port, and that loopback is unreachable', () => {
    const text = banner({ host: '127.0.0.1', port: 8080, tls: undefined, lan: ['192.168.1.7'], node }).join('\n');
    expect(text).toContain('getUserMedia needs a secure context');
    expect(text).toContain('--tls-port');
    expect(text).toContain('adb reverse tcp:8080 tcp:8080');
    expect(text).toContain('pass --host 0.0.0.0 to reach it from a phone');
    // A specific bind is reachable at that address only.
    expect(text).not.toContain('192.168.1.7');
  });
});
