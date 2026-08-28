# Source me instead of `.env`:  . ./tools/runpod/env.sh
#
# Puts RUNPOD_API_KEY into the environment from the login keyring, so the key
# never sits on disk in cleartext. This repo is public; a plaintext .env is one
# `git add -f` away from being published.
#
# One-time enrolment (or after rotating the key at
# https://www.runpod.io/console/user/settings):
#   secret-tool store --label='RunPod API key' service runpod key RUNPOD_API_KEY
# ...then paste the key at the prompt and press Enter. It is not echoed.
#
# Never `echo`/`cat` the key, and never write it back to a file.

RUNPOD_API_KEY="$(secret-tool lookup service runpod key RUNPOD_API_KEY)" || {
  echo "runpod: no key in the keyring. Enrol it with:" >&2
  echo "  secret-tool store --label='RunPod API key' service runpod key RUNPOD_API_KEY" >&2
  unset RUNPOD_API_KEY
  return 1 2>/dev/null || exit 1
}
export RUNPOD_API_KEY
