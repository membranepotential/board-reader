# Source me: exports AWS_* credentials for RunPod's S3-compatible volume API
# from the login keyring (see tools/runpod/env.sh for the pattern and why no
# .env file is ever acceptable in this public repo).
#
#   . ./tools/runpod/s3-env.sh
#   aws s3 ls "s3://$BR_RP_S3_BUCKET/" --endpoint-url "$BR_RP_S3_ENDPOINT"
#
# The bucket IS the network volume id; the endpoint is DC-specific.
AWS_ACCESS_KEY_ID="$(secret-tool lookup service runpod-s3 part access)" || {
  echo "no RunPod S3 access key in the keyring — see .claude/skills/runpod/SKILL.md" >&2
  return 1 2>/dev/null || exit 1
}
AWS_SECRET_ACCESS_KEY="$(secret-tool lookup service runpod-s3 part secret)" || {
  echo "no RunPod S3 secret in the keyring" >&2
  return 1 2>/dev/null || exit 1
}
export AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY
export AWS_DEFAULT_REGION="${BR_RP_S3_REGION:-EUR-IS-1}"
# The user's real ~/.aws (SSO profiles) must not hijack these calls.
export AWS_CONFIG_FILE=/dev/null AWS_SHARED_CREDENTIALS_FILE=/dev/null
export BR_RP_S3_ENDPOINT="${BR_RP_S3_ENDPOINT:-https://s3api-eur-is-1.runpod.io}"
# The bucket IS the volume id. The EU-RO-1 volume (n102g25d6l) was destroyed
# 2026-08-31; there is no default until a volume exists again — set it from
# `runpodctl network-volume list` after recreating (skill runpod §2).
export BR_RP_S3_BUCKET="${BR_RP_S3_BUCKET:-}"
[ -n "$BR_RP_S3_BUCKET" ] || echo "warning: BR_RP_S3_BUCKET unset — no network volume exists; create one and set this to its id" >&2
