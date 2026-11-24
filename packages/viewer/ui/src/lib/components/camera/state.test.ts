import { describe, expect, it } from 'vitest';
import type { CornerQuad, Point } from '$lib/protocol';
import {
  ADOPT_CONSECUTIVE,
  ADOPT_MAX_SHIFT_SQUARES,
  ADOPT_RELABEL_MARGIN,
  AUTO_BACKOFF_MS,
  AUTO_STABILITY,
  AUTO_DETECT_START,
  AUTO_LOST_MS,
  AUTO_SETTLE_MS,
  autoGaveUp,
  backoffFor,
  QUAD_VERIFY_SHIFT,
  quadShift,
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
  cornerOffsets,
  matchCorners,
  type CornerOffsets,
  type RedetectGate,
  noteRedetectMiss,
  noteRedetection,
  REDETECT_BACKOFF_MS,
  REDETECT_INTERVAL_MS,
  REDETECT_START,
  redetectBackoffFor,
  redetectGaveUp,
  shouldRedetect,
  squareWidth,
  toQuad,
  trackingChip,
  withinRateWindow,
  type Activity,
  type Redetect,
  type RedetectOutcome,
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
      { kind: 'detecting', auto: true }
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

describe('the hint line', () => {
  const base = {
    camera: live(),
    activity: IDLE,
    auto: AUTO_DETECT_START,
    tracking: 'locked'
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

import {
  freshFrameClock,
  noteFrameClock,
  STALL_PLAY_MS,
  STALL_REOPEN_MS,
  STALL_REOPEN_MAX_MS
} from './state';

describe('camera stall watchdog', () => {
  it('an advancing clock is healthy and resets every escalation', () => {
    let clock = freshFrameClock(0);
    for (let now = 50, t = 0.05; now < 500; now += 50, t += 0.05) {
      const step = noteFrameClock(clock, t, now, false);
      clock = step.clock;
      expect(step.action).toBe('none');
    }
    expect(clock.reopens).toBe(0);
  });

  it('escalates play -> reopen -> backed-off reopen while the clock is stuck', () => {
    let clock = freshFrameClock(0);
    clock = noteFrameClock(clock, 1.0, 0, false).clock; // first reading

    let step = noteFrameClock(clock, 1.0, STALL_PLAY_MS, false);
    expect(step.action).toBe('play');
    clock = step.clock;

    // play() did nothing; still stalled at the reopen threshold
    step = noteFrameClock(clock, 1.0, STALL_REOPEN_MS, false);
    expect(step.action).toBe('reopen');
    clock = step.clock;
    expect(clock.reopens).toBe(1);
    expect(clock.nextReopenAt).toBe(STALL_REOPEN_MS + STALL_REOPEN_MS);

    // before the backoff expires: quiet, not a reopen storm
    step = noteFrameClock(clock, 1.0, STALL_REOPEN_MS + 100, false);
    expect(step.action).toBe('none');

    // after it: the second reopen, with a doubled backoff
    step = noteFrameClock(clock, 1.0, clock.nextReopenAt, false);
    expect(step.action).toBe('reopen');
    expect(step.clock.nextReopenAt - clock.nextReopenAt).toBe(2 * STALL_REOPEN_MS);
  });

  it('caps the reopen backoff', () => {
    let clock = freshFrameClock(0);
    clock = noteFrameClock(clock, 1.0, 0, false).clock;
    clock = noteFrameClock(clock, 1.0, STALL_PLAY_MS, false).clock; // play, spent
    for (let i = 0; i < 8; i++) {
      const at = Math.max(clock.nextReopenAt, STALL_REOPEN_MS);
      const step = noteFrameClock(clock, 1.0, at, false);
      expect(step.action).toBe('reopen');
      expect(step.clock.nextReopenAt - at).toBeLessThanOrEqual(STALL_REOPEN_MAX_MS);
      clock = step.clock;
    }
  });

  it('a hidden page is never a stall, and returning starts a fresh grace period', () => {
    let clock = freshFrameClock(0);
    clock = noteFrameClock(clock, 1.0, 0, false).clock;
    // hidden through what would have been a stall
    let step = noteFrameClock(clock, 1.0, STALL_REOPEN_MS * 2, true);
    expect(step.action).toBe('none');
    clock = step.clock;
    // just visible again: still no action — the grace period restarted
    step = noteFrameClock(clock, 1.0, STALL_REOPEN_MS * 2 + 100, false);
    expect(step.action).toBe('none');
  });

  it('recovery is proven by the clock advancing, which zeroes the backoff', () => {
    let clock = freshFrameClock(0);
    clock = noteFrameClock(clock, 1.0, 0, false).clock;
    clock = noteFrameClock(clock, 1.0, STALL_REOPEN_MS, false).clock; // reopen #1
    const step = noteFrameClock(clock, 0.1, STALL_REOPEN_MS + 500, false); // new stream
    expect(step.action).toBe('none');
    expect(step.clock.reopens).toBe(0);
    expect(step.clock.nextReopenAt).toBe(0);
  });
});

describe('restored-framing verification', () => {
  it('a restored framing triggers one auto-detection once the scene is still', () => {
    const gate = {
      auto: { fails: 0, nextAt: 0, lostSince: null },
      now: 10_000,
      camera: live('restored', 0),
      activity: IDLE,
      hidden: false,
      stability: 0.95
    };
    expect(shouldAutoDetect(gate)).toBe(true);
    // a detected or operator framing does not — this fires on page open only
    expect(shouldAutoDetect({ ...gate, camera: live('detected', 0) })).toBe(false);
    expect(shouldAutoDetect({ ...gate, camera: live('operator', 0) })).toBe(false);
  });

  it('quadShift is pointwise — a reordering of the same points is a big shift', () => {
    const frame = { width: 1000, height: 1000 };
    const quad: CornerQuad = [
      [100, 100],
      [900, 100],
      [900, 900],
      [100, 900]
    ];
    const nudged = quad.map(([x, y]) => [x + 5, y + 5]) as unknown as CornerQuad;
    expect(quadShift(quad, nudged, frame)).toBeLessThan(QUAD_VERIFY_SHIFT);
    const reordered = [quad[1], quad[2], quad[3], quad[0]] as unknown as CornerQuad;
    expect(quadShift(quad, reordered, frame)).toBeGreaterThan(QUAD_VERIFY_SHIFT);
  });
});

describe('periodic re-detection', () => {
  // A square board, 800 px on a side, so one square is exactly 100 px: every
  // shift below can be read as a fraction of a square by eye.
  const TRACKED: CornerQuad = [
    [200, 100],
    [1000, 100],
    [1000, 900],
    [200, 900]
  ];
  const SQUARE = 100;

  /** `TRACKED` with a per-corner displacement applied — a drifted or offset quad. */
  function moved(dx: number, dy = 0, corners: readonly number[] = [0, 1, 2, 3]): CornerQuad {
    const at = (i: number): Point =>
      corners.includes(i)
        ? [(TRACKED[i] as Point)[0] + dx, (TRACKED[i] as Point)[1] + dy]
        : (TRACKED[i] as Point);
    return [at(0), at(1), at(2), at(3)];
  }

  const gate: RedetectGate = {
    redetect: REDETECT_START,
    now: 60_000,
    camera: live('detected'),
    activity: IDLE,
    streaming: true,
    stability: 1,
    bursting: false,
    portrait: false,
    hidden: false,
    tracking: true
  };

  describe('the gate', () => {
    it('runs on a resting, streaming, framed page', () => {
      expect(shouldRedetect(gate)).toBe(true);
    });

    it('refuses everything the check is not for', () => {
      const cases: readonly [string, Partial<RedetectGate>][] = [
        ['not streaming', { streaming: false }],
        ['portrait', { portrait: true }],
        ['mid-burst', { bursting: true }],
        ['hidden', { hidden: true }],
        ['moving', { stability: AUTO_STABILITY - 0.01 }],
        ['tracking switched off', { tracking: false }],
        ['no camera', { camera: { kind: 'requesting' } }],
        ['never framed', { camera: live('placeholder') }],
        ['spent its failure budget', { redetect: { ...REDETECT_START, fails: 3 } }],
        ['dragging', { activity: { kind: 'dragging', corner: 0 } }],
        ['detecting', { activity: { kind: 'detecting', auto: true } }],
        ['already checking', { activity: { kind: 'verifying' } }],
        ['inside the interval', { redetect: { ...REDETECT_START, nextAt: gate.now + 1 } }]
      ];
      for (const [why, override] of cases) {
        expect(shouldRedetect({ ...gate, ...override }), why).toBe(false);
      }
    });

    it('checks a board framed by hand as readily as a detected one', () => {
      // The convention offset is exactly what the baseline is for, so a
      // finger-dragged framing is not excluded from being checked.
      expect(shouldRedetect({ ...gate, camera: live('operator') })).toBe(true);
    });

    it('never baselines from a RESTORED framing (review, 0ebbbd2)', () => {
      // Last session's quad over this session's world can be a full rank out —
      // that is what `shouldAutoDetect` verifies one for — and R0 is permanent.
      // Were a stale quad allowed to baseline (as it would be if that
      // verification had burned its budget against a slow node), the rank-off
      // framing becomes this session's definition of correct and genuine drift
      // reads as zero for ever. The check starts once it is re-origined.
      expect(shouldRedetect({ ...gate, camera: live('restored') })).toBe(false);
      expect(shouldRedetect({ ...gate, camera: live('detected') })).toBe(true);
    });

    it('spaces successful checks by the interval', () => {
      const after = noteRedetection({
        redetect: REDETECT_START,
        now: gate.now,
        detected: TRACKED,
        tracked: TRACKED
      }).redetect;
      expect(after.nextAt).toBe(gate.now + REDETECT_INTERVAL_MS);
      expect(shouldRedetect({ ...gate, redetect: after })).toBe(false);
      expect(
        shouldRedetect({ ...gate, redetect: after, now: gate.now + REDETECT_INTERVAL_MS })
      ).toBe(true);
    });

    it('backs off on consecutive misses and then stops asking', () => {
      // A page pointed at a board the detector cannot see would otherwise post a
      // full-frame JPEG every ten seconds for the whole session.
      let state = REDETECT_START;
      const gaps: number[] = [];
      let now = 0;
      for (let i = 0; i < REDETECT_BACKOFF_MS.length; i++) {
        state = noteRedetectMiss(state, now);
        gaps.push(state.nextAt - now);
        now = state.nextAt;
      }
      expect(gaps).toEqual([...REDETECT_BACKOFF_MS]);
      expect(state.fails).toBe(REDETECT_BACKOFF_MS.length);
      expect(redetectGaveUp(state)).toBe(true);
      expect(shouldRedetect({ ...gate, redetect: state, now })).toBe(false);
      expect(redetectBackoffFor(99)).toBe(REDETECT_BACKOFF_MS.at(-1));
    });

    it('gives the budget back the moment the detector answers at all', () => {
      const spent: Redetect = { ...REDETECT_START, fails: 2 };
      const { redetect } = noteRedetection({
        redetect: spent,
        now: 0,
        detected: TRACKED,
        tracked: TRACKED
      });
      expect(redetect.fails).toBe(0);
      expect(redetectGaveUp(redetect)).toBe(false);
      // And a deliberate re-framing starts the whole thing over — the page puts
      // `REDETECT_START` back on every one of those paths.
      expect(redetectGaveUp(REDETECT_START)).toBe(false);
    });
  });

  describe('the square-width unit', () => {
    it('is the mean side over eight, not a fraction of the frame', () => {
      expect(squareWidth(TRACKED)).toBeCloseTo(SQUARE, 6);
    });

    it('is NOT what quadShift measures — the reason this has its own unit', () => {
      // 0.45 square of shift on this board is 45 px, which the frame-diagonal
      // convention calls 0.031 — a hair over QUAD_VERIFY_SHIFT. The two numbers
      // are not interchangeable and the report says so explicitly.
      const shifted = moved(45);
      expect(quadShift(TRACKED, shifted, FRAME)).toBeLessThan(2 * QUAD_VERIFY_SHIFT);
      expect(quadShift(TRACKED, shifted, FRAME)).toBeGreaterThan(QUAD_VERIFY_SHIFT);
    });
  });

  describe('corner correspondence', () => {
    it('matches a detection back onto the tracked slots whatever order it arrived in', () => {
      const rotated: CornerQuad = [TRACKED[2], TRACKED[3], TRACKED[0], TRACKED[1]];
      const match = matchCorners(rotated, TRACKED);
      expect(match.aligned).toEqual(TRACKED);
      expect(match.cost).toBe(0);
      expect(match.margin).toBe(Infinity);
    });

    it('matches a reversed winding too', () => {
      const flipped: CornerQuad = [TRACKED[0], TRACKED[3], TRACKED[2], TRACKED[1]];
      expect(matchCorners(flipped, TRACKED).aligned).toEqual(TRACKED);
    });

    it('leaves the runner-up most of a board away, which is what the margin checks', () => {
      const match = matchCorners(moved(20), TRACKED);
      expect(match.cost).toBeCloseTo(20, 6);
      expect(match.margin).toBeGreaterThan(ADOPT_RELABEL_MARGIN);
    });

    it('refuses to BASELINE from an ambiguous correspondence', () => {
      // A quad small enough that every order is about equally wrong: no
      // correspondence is safe, so nothing may become the session's convention.
      const tiny: CornerQuad = [
        [600, 490],
        [610, 490],
        [610, 500],
        [600, 500]
      ];
      const { redetect, outcome } = noteRedetection({
        redetect: REDETECT_START,
        now: 1000,
        detected: tiny,
        tracked: TRACKED
      });
      expect(outcome.kind).toBe('ambiguous');
      expect(redetect.samples).toEqual([]);
      expect(redetect.residual).toBeNull();
      expect(redetect.nextAt).toBe(1000 + REDETECT_INTERVAL_MS);
    });
  });

  describe('the baseline', () => {
    it('spends the first three detections measuring, and takes their median', () => {
      // A 30 px convention offset with one bad sample in the middle. The median
      // ignores the outlier; a mean would carry 6.7 px of it forever.
      let state = REDETECT_START;
      const samples = [30, 130, 32];
      const kinds: string[] = [];
      samples.forEach((dx, i) => {
        const step = noteRedetection({
          redetect: state,
          now: i * REDETECT_INTERVAL_MS,
          detected: moved(dx),
          tracked: TRACKED
        });
        state = step.redetect;
        kinds.push(step.outcome.kind);
      });
      expect(kinds).toEqual(['baselining', 'baselining', 'baselining']);
      expect(state.residual).toEqual([
        [32, 0],
        [32, 0],
        [32, 0],
        [32, 0]
      ]);
      expect(state.samples).toEqual([]);
    });

    it('is not advanced by a miss', () => {
      let state = noteRedetection({
        redetect: REDETECT_START,
        now: 0,
        detected: moved(30),
        tracked: TRACKED
      }).redetect;
      state = noteRedetectMiss(state, REDETECT_INTERVAL_MS);
      expect(state.samples).toHaveLength(1);
      expect(state.residual).toBeNull();
    });
  });

  /** Run a sequence of detections (`null` is a detector miss) from a given start. */
  function replay(
    detections: readonly (CornerQuad | null)[],
    start: Redetect = REDETECT_START
  ): { readonly state: Redetect; readonly outcomes: readonly RedetectOutcome[] } {
    let state = start;
    const outcomes: RedetectOutcome[] = [];
    detections.forEach((detected, i) => {
      const now = i * REDETECT_INTERVAL_MS;
      if (detected === null) {
        state = noteRedetectMiss(state, now);
        return;
      }
      const step = noteRedetection({ redetect: state, now, detected, tracked: TRACKED });
      state = step.redetect;
      outcomes.push(step.outcome);
    });
    return { state, outcomes };
  }

  describe('adoption', () => {
    it('never fires on a constant convention offset, however large', () => {
      // The hand-framed sessions: the detector's line sits a fixed 30 px from the
      // operator's — 0.3 square, most of the way to the threshold — for ever. An
      // absolute threshold would false-alarm on every one of them.
      const { outcomes } = replay(Array.from({ length: 12 }, () => moved(30)));
      expect(outcomes.slice(3).every((o) => o.kind === 'steady')).toBe(true);
      for (const outcome of outcomes.slice(3)) {
        if (outcome.kind === 'steady') expect(outcome.shift).toBeCloseTo(0, 6);
      }
    });

    it('adopts on the SECOND consecutive over-threshold reading of a drift ramp', () => {
      // The tracked quad walks; the detector keeps reporting the true board, so
      // the gap grows. 30 px of that is the convention offset, the rest is drift.
      const ramp = [30, 30, 30, 40, 60, 85, 110].map((dx) => moved(dx));
      const { outcomes } = replay(ramp);
      expect(outcomes.map((o) => o.kind)).toEqual([
        'baselining',
        'baselining',
        'baselining',
        'steady', // 0.10 square
        'steady', // 0.30 square
        'arming', // 0.55 square — over, but only once
        'adopt' //   0.80 square — twice in a row
      ]);
    });

    it('adopts the detector quad in the OPERATOR’s convention, not the detector’s', () => {
      const { outcomes } = replay([30, 30, 30, 85, 110].map((dx) => moved(dx)));
      const adopted = outcomes.at(-1);
      expect(adopted?.kind).toBe('adopt');
      if (adopted?.kind !== 'adopt') return;
      // 110 px detected minus the 30 px convention offset: the corners the
      // operator would have dragged, on the board the detector actually found.
      expect(adopted.quad).toEqual(moved(80));
      expect(adopted.shift).toBeCloseTo(0.8, 6);
    });

    it('starts the count again after adopting, but keeps R0', () => {
      const { state } = replay([30, 30, 30, 85, 110].map((dx) => moved(dx)));
      expect(state.over).toBe(0);
      expect(state.residual).toEqual([
        [30, 0],
        [30, 0],
        [30, 0],
        [30, 0]
      ]);
      // And having adopted, the very next reading of the same gap is steady
      // again only once the tracker has actually been moved — the pure layer
      // just counts, so a still-drifted reading arms afresh rather than adopting.
      const { outcomes } = replay([moved(110)], state);
      expect(outcomes.map((o) => o.kind)).toEqual(['arming']);
    });

    it('treats a miss as neutral: no progress, no reset', () => {
      const ramp = [moved(30), moved(30), moved(30), moved(85), null, moved(110)];
      const { outcomes } = replay(ramp);
      // The miss did not disarm the counter, so the reading after it still
      // adopts — misses cluster on the drifted frames themselves.
      expect(outcomes.map((o) => o.kind)).toEqual([
        'baselining',
        'baselining',
        'baselining',
        'arming',
        'adopt'
      ]);
    });

    it('disarms on a reading back inside the threshold', () => {
      const { outcomes } = replay([30, 30, 30, 85, 35, 85, 40].map((dx) => moved(dx)));
      expect(outcomes.map((o) => o.kind)).toEqual([
        'baselining',
        'baselining',
        'baselining',
        'arming',
        'steady',
        'arming',
        'steady'
      ]);
    });

    it('refuses to conclude ANYTHING from a quad it cannot relabel — it only delays', () => {
      // The last guard before a8 could move. It needs a detection whose corners do
      // NOT pair off decisively while the baselined signal still looks like
      // ordinary drift, which takes an outsized convention offset to build: the
      // margin is not scale-free (for a translation d of a board of side S the
      // runner-up costs S + d), so with a realistic R0 of 17-33 px the reposition
      // cap always fires first.
      const offset: CornerOffsets = [
        [120, 0],
        [120, 0],
        [120, 0],
        [120, 0]
      ];
      const armed: Redetect = {
        ...REDETECT_START,
        residual: offset,
        over: ADOPT_CONSECUTIVE - 1
      };
      const { redetect, outcome } = noteRedetection({
        redetect: armed,
        now: 0,
        detected: moved(280), // 1.6 squares past the offset; margin 3.9, not decisive
        tracked: TRACKED
      });
      expect(outcome.kind).toBe('ambiguous');
      if (outcome.kind === 'ambiguous') {
        expect(outcome.margin).toBeLessThan(ADOPT_RELABEL_MARGIN);
      }
      // Every counter untouched: such a reading can delay an adoption, never
      // cause one and never cancel one (review, 0ebbbd2).
      expect(redetect.over).toBe(armed.over);
      expect(redetect.residual).toEqual(offset);
      expect(redetect.suspects).toBe(0);
      // The next decisive reading of the same drift is the one that acts.
      const decisive = noteRedetection({
        redetect,
        now: REDETECT_INTERVAL_MS,
        detected: moved(120 + 0.8 * SQUARE), // 0.8 square of drift, margin 5.0
        tracked: TRACKED
      });
      expect(decisive.outcome.kind).toBe('adopt');
    });

    it('refuses to adopt a reposition, and says so once per episode', () => {
      // Beyond ADOPT_MAX_SHIFT_SQUARES it is not drift — the phone or the board
      // moved. Adopting would quietly re-crop around a board nobody framed.
      const far = moved(30 + (ADOPT_MAX_SHIFT_SQUARES + 0.5) * SQUARE);
      const { state, outcomes } = replay([moved(30), moved(30), moved(30), far, far, far]);
      const suspects = outcomes.slice(3);
      expect(suspects.map((o) => o.kind)).toEqual(['suspect', 'suspect', 'suspect']);
      expect(suspects.map((o) => (o.kind === 'suspect' ? o.repeats : 0))).toEqual([1, 2, 3]);
      // Said once, not three times: `warn` is what the page shows a banner for.
      expect(suspects.map((o) => (o.kind === 'suspect' ? o.warn : true))).toEqual([
        true,
        false,
        false
      ]);
      expect(state.over).toBe(0);
    });

    it('does not say it again while a quad oscillates across the cap', () => {
      // suspect / arming / suspect is one episode, not two: only a reading that
      // is genuinely back in place ends it (review, 0ebbbd2).
      const far = moved(30 + (ADOPT_MAX_SHIFT_SQUARES + 0.5) * SQUARE);
      const near = moved(30 + 0.5 * SQUARE); // over the adopt threshold, under the cap
      const settled = moved(30);
      const { outcomes } = replay([
        moved(30),
        moved(30),
        moved(30),
        far,
        near,
        far,
        settled,
        far
      ]);
      const said = outcomes.map((o) => (o.kind === 'suspect' ? o.warn : null));
      expect(outcomes.slice(3).map((o) => o.kind)).toEqual([
        'suspect',
        'arming',
        'suspect',
        'steady',
        'suspect'
      ]);
      // Warned, silent through the oscillation, then warned again for the NEW
      // episode that follows a genuinely steady reading.
      expect(said.slice(3)).toEqual([true, null, false, null, true]);
    });

    it('does not let a reposition run bleed into the adoption count', () => {
      const far = moved(30 + (ADOPT_MAX_SHIFT_SQUARES + 0.5) * SQUARE);
      const { outcomes } = replay([moved(30), moved(30), moved(30), far, far, moved(85)]);
      // Two suspects then one genuinely-drifted reading: arming, not adopting.
      expect(outcomes.at(-1)?.kind).toBe('arming');
    });

    it('measures the WORST corner, so one corner walking is enough', () => {
      // Take 5's shape: a1 reached 82 px while h1 moved 27. A mean would have
      // called that a third of a square and stayed silent.
      const oneCorner = (dx: number): CornerQuad => moved(dx, 0, [3]);
      const { outcomes } = replay([oneCorner(0), oneCorner(0), oneCorner(0), oneCorner(80), oneCorner(85)]);
      expect(outcomes.map((o) => o.kind)).toEqual([
        'baselining',
        'baselining',
        'baselining',
        'arming',
        'adopt'
      ]);
    });
  });
});
