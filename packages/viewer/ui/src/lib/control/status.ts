/** What the control page says about time, as an operator reads it. */

/** `2.3 s ago`, `4 min ago`: an age in seconds (on the node's clock). */
export function formatAge(seconds: number | null): string {
  if (seconds === null) return 'never';
  if (seconds < 60) return `${seconds.toFixed(1)} s ago`;
  return `${Math.round(seconds / 60)} min ago`;
}
