import type { HandleServerError } from '@sveltejs/kit/hooks';

/**
 * Log real server errors only: those thrown by our code or what it calls. SvelteKit's
 * own errors (a 404, a 405) are the browser's problem, and every browser asks for
 * `/favicon.ico` on every page load, so logging them filled the viewer's log with
 * lines about nothing.
 */
export const handleError: HandleServerError = ({ kind, error }) => {
  if (kind === 'unknown') console.error(error);
};
