import { describe, expect, it } from 'vitest';
import type { CornerQuad } from '$lib/protocol';
import {
  AUTO_BACKOFF_MS,
  AUTO_DETECT_START,
  AUTO_LOST_MS,
  AUTO_SETTLE_MS,
  autoGaveUp,
  backoffFor,
  calibrateBlockReason,
  denormaliseQuad,
  hintFor,
  isFramed,
  noteTracking,
  normaliseQuad,
  PANEL_START,
  parseStoredFraming,
  parseStoredPanel,
  placeholderQuad,
  placePanel,
  ratePerSecond,
  rebaseQuad,
  serialiseFraming,
  shouldAutoDetect,
  toQuad,
  trackingChip,
  withinRateWindow,
  type Activity,
  type AutoDetect,
  type CameraState,
  type Framing
} from './state';

const FRAME = { width: 1280, height: 720 } as const;
const QUAD: CornerQuad = [
  [100, 200],
  [900, 210],
  [910, 640],
  [110, 630]
];

function framing(origin: Framing['origin']): Framing {
  return { quad: QUAD, rot: 0, origin };
}

function live(origin: Framing['origin'] = 'operator', liveAt = 0): CameraState {
  return { kind: 'live', frame: FRAME, framing: framing(origin), liveAt };
}

const IDLE: Activity = { kind: 'idle' };

describe('quads', () => {
  it('places the placeholder in the middle of the frame', () => {
    const quad = placeholderQuad(FRAME);
    expect(quad).toHaveLength(4);
    for (const [x, y] of quad) {
      expect(x).toBeGreaterThan(0);
      expect(x).toBeLessThan(FRAME.width);
      expect(y).toBeGreaterThan(0);
      expect(y).toBeLessThan(FRAME.height);
    }
  });

  it('normalise and denormalise round-trip', () => {
    expect(denormaliseQuad(normaliseQuad(QUAD, FRAME), FRAME)).toEqual(QUAD);
  });

  it('rebases a quad onto a frame that swapped its axes', () => {
    const turned = { width: 720, height: 1280 } as const;
    const moved = rebaseQuad(QUAD, FRAME, turned);
    // The same FRACTION of the frame, which is the whole point of rebasing.
    expect(normaliseQuad(moved, turned)).toEqual(normaliseQuad(QUAD, FRAME));
  });

  it('narrows only a list of exactly four points', () => {
    expect(toQuad([])).toBeNull();
    expect(toQuad([[0, 0], [1, 0], [1, 1]])).toBeNull();
    expect(toQuad([[0, 0], [1, 0], [1, 1], [0, 1]])).toEqual([[0, 0], [1, 0], [1, 1], [0, 1]]);
  });
});

describe('stored framing', () => {
  it('round-trips through fractions, not pixels', () => {
    const stored = serialiseFraming({ quad: QUAD, rot: 2, origin: 'operator' }, FRAME);
    const restored = parseStoredFraming(stored, FRAME);
    expect(restored?.quad).toEqual(QUAD);
    expect(restored?.rot).toBe(2);
  });

  it('restores against the CURRENT frame, so a resolution change is absorbed', () => {
    const stored = serialiseFraming({ quad: QUAD, rot: 0, origin: 'operator' }, FRAME);
    const half = { width: 640, height: 360 } as const;
    expect(parseStoredFraming(stored, half)?.quad).toEqual(
      QUAD.map(([x, y]) => [x / 2, y / 2])
    );
  });

  it('counts as a framing, so automatic detection stands down', () => {
    const stored = serialiseFraming({ quad: QUAD, rot: 0, origin: 'operator' }, FRAME);
    const restored = parseStoredFraming(stored, FRAME);
    expect(restored?.origin).toBe('restored');
    expect(restored !== null && isFramed(restored)).toBe(true);
  });

  it('refuses anything that is not four numeric pairs', () => {
    expect(parseStoredFraming(null, FRAME)).toBeNull();
    expect(parseStoredFraming('not json', FRAME)).toBeNull();
    expect(parseStoredFraming('{"pts":[]}', FRAME)).toBeNull();
    expect(parseStoredFraming('{"pts":[[0,0],[1,0],[1,1]]}', FRAME)).toBeNull();
    expect(parseStoredFraming('{"pts":[[0,0],[1,0],[1,1],["x",1]]}', FRAME)).toBeNull();
    expect(parseStoredFraming('{"pts":[[0,0],[1,0],[1,1],[0,null]]}', FRAME)).toBeNull();
  });

  it('folds a nonsense rotation into the four legal ones', () => {
    expect(parseStoredFraming('{"pts":[[0,0],[1,0],[1,1],[0,1]],"rot":7}', FRAME)?.rot).toBe(3);
    expect(parseStoredFraming('{"pts":[[0,0],[1,0],[1,1],[0,1]],"rot":"2"}', FRAME)?.rot).toBe(0);
  });

  it('a placeholder is not a framing', () => {
    expect(isFramed({ quad: QUAD, rot: 0, origin: 'placeholder' })).toBe(false);
  });
});

describe('automatic detection', () => {
  const settled = AUTO_SETTLE_MS + 1;

  it('waits out the settle delay even at a perfect stability', () => {
    // MotionMeter reports 1.0 on its FIRST frame by construction, so without the
    // settle delay this fires before autoexposure and autofocus have settled.
    expect(
      shouldAutoDetect({
        auto: AUTO_DETECT_START,
        now: AUTO_SETTLE_MS - 1,
        camera: live('placeholder'),
        activity: IDLE,
        hidden: false,
        stability: 1
      })
    ).toBe(false);
  });

  it('fires on first sight of an unframed board once settled and still', () => {
    expect(
      shouldAutoDetect({
        auto: AUTO_DETECT_START,
        now: settled,
        camera: live('placeholder'),
        activity: IDLE,
        hidden: false,
        stability: 1
      })
    ).toBe(true);
  });

  it('demands a stability stricter than merely "not moving"', () => {
    expect(
      shouldAutoDetect({
        auto: AUTO_DETECT_START,
        now: settled,
        camera: live('placeholder'),
        activity: IDLE,
        hidden: false,
        stability: 0.85
      })
    ).toBe(false);
  });

  it('stands down while anything else owns the camera, or the page is hidden', () => {
    const base = {
      auto: AUTO_DETECT_START,
      now: settled,
      camera: live('placeholder'),
      hidden: false,
      stability: 1
    } as const;
    for (const activity of [
      { kind: 'dragging', corner: 0 },
      { kind: 'detecting', auto: true },
      { kind: 'calibrating' }
    ] as const satisfies readonly Activity[]) {
      expect(shouldAutoDetect({ ...base, activity })).toBe(false);
    }
    expect(shouldAutoDetect({ ...base, activity: IDLE, hidden: true })).toBe(false);
  });

  it('leaves a framed board alone until the loss is SUSTAINED', () => {
    // PlaneTracker.lose() self-heals, so `lost` fires on essentially every move of
    // a real game; re-detecting on the raw flag would re-detect mid-game.
    const now = settled + AUTO_LOST_MS * 2;
    const brief: AutoDetect = { ...AUTO_DETECT_START, lostSince: now - 1 };
    const sustained: AutoDetect = { ...AUTO_DETECT_START, lostSince: now - AUTO_LOST_MS };
    const base = { now, camera: live('operator'), activity: IDLE, hidden: false, stability: 1 } as const;
    expect(shouldAutoDetect({ ...base, auto: AUTO_DETECT_START })).toBe(false);
    expect(shouldAutoDetect({ ...base, auto: brief })).toBe(false);
    expect(shouldAutoDetect({ ...base, auto: sustained })).toBe(true);
  });

  it('honours the back-off window', () => {
    expect(
      shouldAutoDetect({
        auto: { fails: 1, nextAt: settled + 1000, lostSince: null },
        now: settled,
        camera: live('placeholder'),
        activity: IDLE,
        hidden: false,
        stability: 1
      })
    ).toBe(false);
  });

  it('gives up after four attempts, and the gaps grow', () => {
    expect(autoGaveUp(AUTO_DETECT_START)).toBe(false);
    expect(autoGaveUp({ ...AUTO_DETECT_START, fails: 3 })).toBe(false);
    expect(autoGaveUp({ ...AUTO_DETECT_START, fails: 4 })).toBe(true);
    // After the first miss the next attempt is 8 s out, then 16, then 32.
    expect([0, 1, 2, 3, 4].map(backoffFor)).toEqual([...AUTO_BACKOFF_MS, 32000]);
    expect(
      shouldAutoDetect({
        auto: { fails: AUTO_BACKOFF_MS.length, nextAt: 0, lostSince: null },
        now: settled,
        camera: live('placeholder'),
        activity: IDLE,
        hidden: false,
        stability: 1
      })
    ).toBe(false);
  });

  it('starts the loss clock once and clears it on recovery', () => {
    const lost = noteTracking(AUTO_DETECT_START, 'lost', 100);
    expect(lost.lostSince).toBe(100);
    expect(noteTracking(lost, 'lost', 500).lostSince).toBe(100);
    expect(noteTracking(lost, 'locked', 500).lostSince).toBeNull();
    expect(noteTracking(AUTO_DETECT_START, 'off', 500)).toBe(AUTO_DETECT_START);
  });
});

describe('calibrate guard', () => {
  // No reason mentions stability. A stability gate shipped here once and made
  // calibrate permanently unreachable on the one device the page is for: the meter is
  // mirrored value-for-value with the MOUNTED Python edge, so handheld micro-shake
  // sits under MOVING_BELOW essentially always. DESIGN D3 — a hint, never a gate.
  it('names its reason, in the words the operator is shown', () => {
    expect(calibrateBlockReason({ camera: live(), activity: { kind: 'calibrating' } })).toBe(
      'calibrating…'
    );
    expect(calibrateBlockReason({ camera: { kind: 'requesting' }, activity: IDLE })).toBe(
      'the camera is not running'
    );
    // Live and idle is enough. Nothing about how still the phone is being held
    // reaches this function at all — that is what the missing input means.
    expect(calibrateBlockReason({ camera: live(), activity: IDLE })).toBeNull();
  });

  it('blocks under a detection, which owns the quad and ends by clearing `activity`', () => {
    // `/detect` takes 2-4 s on a miss and its `finally` sets activity back to idle
    // unconditionally. A calibration allowed to start underneath one would have that
    // fire mid-burst, resuming the forwarder into a half-rebuilt bank.
    for (const auto of [true, false]) {
      expect(calibrateBlockReason({ camera: live(), activity: { kind: 'detecting', auto } })).toBe(
        'looking for the board…'
      );
    }
  });

  it('feeds the hint line, so a disabled control says why on a screen with no hover', () => {
    const block = calibrateBlockReason({
      camera: live(),
      activity: { kind: 'detecting', auto: true }
    });
    expect(
      hintFor({
        camera: live(),
        activity: { kind: 'detecting', auto: true },
        auto: AUTO_DETECT_START,
        tracking: 'locked',
        streaming: false,
        calibrateBlock: block
      })?.key
      // The detecting hint outranks it — same fact, better sentence.
    ).toBe('detecting');
  });
});

describe('the hint line', () => {
  const base = {
    camera: live(),
    activity: IDLE,
    auto: AUTO_DETECT_START,
    tracking: 'locked',
    streaming: false,
    calibrateBlock: null
  } as const;

  it('says nothing when there is nothing to say', () => {
    expect(hintFor(base)).toBeNull();
  });

  it('offers the recovery link once detection has given up', () => {
    const hint = hintFor({ ...base, auto: { fails: 4, nextAt: 0, lostSince: null } });
    expect(hint?.action).toBe('look-again');
  });

  it('ranks give-up over detecting over no-camera over unframed over lost', () => {
    const gaveUp = { fails: 4, nextAt: 0, lostSince: null };
    expect(
      hintFor({ ...base, auto: gaveUp, activity: { kind: 'detecting', auto: false } })?.key
    ).toBe('giveup');
    expect(
      hintFor({ ...base, activity: { kind: 'detecting', auto: false }, camera: { kind: 'idle' } })
        ?.key
    ).toBe('detecting');
    expect(hintFor({ ...base, camera: { kind: 'requesting' } })?.key).toBe('nocam');
    expect(hintFor({ ...base, camera: live('placeholder'), tracking: 'lost' })?.key).toBe('finding');
    expect(hintFor({ ...base, tracking: 'lost' })?.key).toBe('lost');
  });

  it('stays quiet about calibrate while streaming', () => {
    const blocked = { ...base, calibrateBlock: 'the board is moving' } as const;
    expect(hintFor(blocked)?.key).toBe('calib:the board is moving');
    expect(hintFor({ ...blocked, streaming: true })).toBeNull();
  });

  it('says nothing at all when the camera is blocked — the explainer owns the screen', () => {
    expect(
      hintFor({ ...base, camera: { kind: 'blocked', reason: { kind: 'insecure' } } })
    ).toBeNull();
  });
});

describe('chips', () => {
  it('gives every tracking state a tone', () => {
    expect(trackingChip('off').tone).toBe('neutral');
    expect(trackingChip('idle').tone).toBe('neutral');
    expect(trackingChip('seeded').tone).toBe('warn');
    expect(trackingChip('locked').tone).toBe('ok');
    expect(trackingChip('lost').tone).toBe('bad');
  });
});

describe('send rate', () => {
  it('measures (n-1)/span, so a 1 fps keepalive reads as 1.0', () => {
    expect(ratePerSecond([0, 1000, 2000, 3000], true)).toBeCloseTo(1.0, 6);
  });

  it('is zero when not streaming and nothing has been sent', () => {
    expect(ratePerSecond([], false)).toBe(0);
  });

  it('drops stamps that have left the window', () => {
    expect(withinRateWindow([0, 2000, 4000], 4000)).toEqual([2000, 4000]);
  });
});

describe('the position panel', () => {
  const view = { innerWidth: 800, innerHeight: 400 } as const;

  it('starts hidden — it used to cover the picture the operator was aiming', () => {
    expect(PANEL_START.shown).toBe(false);
    // A phone that already chose still wins: this is a first-visit default only.
    expect(parseStoredPanel('{"left":10,"top":20,"width":300,"shown":true}').shown).toBe(true);
  });

  it('parks top-left, clear of the chips and of the overlay buttons on the right', () => {
    const placed = placePanel({ left: null, top: null, width: 220, shown: true }, view, 250);
    expect(placed.left).toBe(12);
    expect(placed.top).toBe(46);
  });

  it('clamps a panel parked off-screen back into the viewport', () => {
    const placed = placePanel({ left: 5000, top: 5000, width: 220, shown: true }, view, 250);
    expect(placed.left).toBe(800 - 220 - 4);
    expect(placed.top).toBe(400 - 250 - 4);
  });

  it('bounds the width — the one degree of freedom the gesture has', () => {
    expect(placePanel({ left: 0, top: 0, width: 10, shown: true }, view, 100).width).toBe(140);
    expect(placePanel({ left: 0, top: 0, width: 9000, shown: true }, view, 100).width).toBe(520);
  });

  it('falls back to the defaults on anything unreadable', () => {
    expect(parseStoredPanel(null).width).toBe(220);
    expect(parseStoredPanel('{').shown).toBe(false);
    expect(parseStoredPanel('{"left":10,"top":20,"width":300,"shown":false}')).toEqual({
      left: 10,
      top: 20,
      width: 300,
      shown: false
    });
  });
});
