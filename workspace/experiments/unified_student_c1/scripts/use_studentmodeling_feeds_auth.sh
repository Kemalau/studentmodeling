#!/usr/bin/env bash

# Every controller in the unified State8 pipeline must resolve the same Azure
# CLI cache. Prefer the dedicated cache when it is healthy. If it has expired
# but the user's default cache has a valid token for the exact Feeds
# subscription, use that default cache consistently instead of forcing every
# watcher onto a stale dedicated cache. Never inherit another project's cache.
studentmodeling_subscription=b6dc87f3-c479-49c8-8cb5-7896da3ff895
# The original dedicated cache was observed returning a transient concatenated
# MSAL payload (JSON "Extra data") while many long-lived controllers polled it
# during device-code refresh.  Use a fresh project-only cache so one bounded
# login owner can establish a clean token store; callers may still override it
# explicitly for recovery or tests.
studentmodeling_dedicated_cache="${STUDENTMODELING_FEEDS_AZURE_CONFIG_DIR:-/home/v-yuejiali/.azure-studentmodeling-feeds-v2}"
studentmodeling_auth_state=/tmp/studentmodeling-feeds-auth-choice.tsv
studentmodeling_auth_lock=/tmp/studentmodeling-feeds-auth-choice.lock
studentmodeling_now="$(date +%s)"
studentmodeling_choice=""

# Do not let dozens of long-lived controllers concurrently refresh and rewrite
# MSAL caches.  One process probes, then all watchers reuse that cache choice
# for sixty seconds.  A failed probe deliberately pins the dedicated cache so
# callers fail/retry without touching the user's default cache again.
exec 217>"$studentmodeling_auth_lock"
flock -w 35 217 || true
if [[ -s "$studentmodeling_auth_state" ]]; then
  read -r studentmodeling_saved_at studentmodeling_choice \
    < "$studentmodeling_auth_state" || studentmodeling_choice=""
  if [[ ! "$studentmodeling_saved_at" =~ ^[0-9]+$ ]] \
      || (( studentmodeling_now - studentmodeling_saved_at >= 60 )); then
    studentmodeling_choice=""
  fi
fi
if [[ -z "$studentmodeling_choice" ]]; then
  if AZURE_CONFIG_DIR="$studentmodeling_dedicated_cache" timeout 30 az account get-access-token \
      --subscription "$studentmodeling_subscription" --query expiresOn -o tsv \
      >/dev/null 2>&1; then
    studentmodeling_choice=dedicated
  # Equivalent fallback command: env -u AZURE_CONFIG_DIR az account get-access-token
  # Keep the outer timeout so a stale default cache cannot pin every watcher.
  elif env -u AZURE_CONFIG_DIR timeout 30 az account get-access-token \
      --subscription "$studentmodeling_subscription" --query expiresOn -o tsv \
      >/dev/null 2>&1; then
    studentmodeling_choice=default
  else
    studentmodeling_choice=unhealthy
  fi
  # This helper is sourced into long-lived parent shells. BASHPID therefore
  # need not be unique across concurrent sources, and a fixed partial name can
  # let one watcher move another watcher's file.  Create the temporary file
  # only after holding the cross-process lock and let mktemp provide a unique
  # name before the atomic replace.
  studentmodeling_auth_tmp="$(mktemp "${studentmodeling_auth_state}.partial.XXXXXX")"
  printf '%s\t%s\n' "$studentmodeling_now" "$studentmodeling_choice" \
    > "$studentmodeling_auth_tmp"
  mv "$studentmodeling_auth_tmp" "$studentmodeling_auth_state"
fi
flock -u 217

if [[ "$studentmodeling_choice" == default ]]; then
  unset AZURE_CONFIG_DIR
else
  export AZURE_CONFIG_DIR="$studentmodeling_dedicated_cache"
fi
export STUDENTMODELING_FEEDS_AUTH_HEALTH="$studentmodeling_choice"
unset studentmodeling_subscription studentmodeling_dedicated_cache
unset studentmodeling_auth_state studentmodeling_auth_lock studentmodeling_now
unset studentmodeling_auth_tmp studentmodeling_saved_at studentmodeling_choice
