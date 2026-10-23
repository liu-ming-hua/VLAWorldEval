#!/usr/bin/env bash
# Shared safe source acquisition. Source this after setting ROOT_DIR.

ensure_source() {
  local source_dir="$1" remote="$2" revision="$3"
  if [[ ! "${revision}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "Source revision must be an exact 40-character commit" >&2
    return 1
  fi
  if [[ -e "${source_dir}" || -L "${source_dir}" ]]; then
    if [[ ! -d "${source_dir}/.git" ]]; then
      echo "Existing source is not a Git checkout: ${source_dir}; not deleting it" >&2
      return 1
    fi
    if [[ "$(git -C "${source_dir}" rev-parse HEAD)" != "${revision}" ]]; then
      echo "Source revision mismatch in ${source_dir}; not changing your checkout" >&2
      return 1
    fi
    # Locally reviewed compatibility patches may be intentional. Preserve them
    # and expose their presence instead of resetting or falsely claiming clean.
    if [[ -n "$(git -C "${source_dir}" status --porcelain)" ]]; then
      echo "WARNING: preserving local source modifications in ${source_dir}" >&2
    fi
    return 0
  fi
  local parent staging
  parent="$(dirname "${source_dir}")"
  mkdir -p "${parent}" || return 1
  # Sibling staging keeps publication on the same filesystem. Never expose a
  # partly acquired checkout at the model's configured source path. Failures
  # retain their fresh staging directory for inspection; retrying can start a
  # new transaction without resetting/removing any earlier checkout.
  staging="$(mktemp -d "${parent}/.source-fetch.XXXXXX")" || return 1
  if ! GIT_LFS_SKIP_SMUDGE=1 git -c http.proxy= -c https.proxy= clone --no-checkout \
      --filter=blob:none "${remote}" "${staging}/checkout" ||
     ! GIT_LFS_SKIP_SMUDGE=1 git -c http.proxy= -c https.proxy= -C "${staging}/checkout" checkout --detach "${revision}" ||
     ! GIT_LFS_SKIP_SMUDGE=1 git -c http.proxy= -c https.proxy= -C "${staging}/checkout" submodule update --init --recursive; then
    echo "Source acquisition failed; retained staging for inspection: ${staging}" >&2
    return 1
  fi
  # GNU mv's no-clobber option can return success without moving. Inspect the
  # staging path too: a concurrent installer/user must never be overwritten.
  if ! mv -T -n -- "${staging}/checkout" "${source_dir}" || [[ -e "${staging}/checkout" ]]; then
    echo "Source publication refused; destination may have appeared. Retained: ${staging}" >&2
    return 1
  fi
  rmdir -- "${staging}"
}
