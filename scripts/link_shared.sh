#!/usr/bin/env bash
# scripts/link_shared.sh -- link this repository's data/ (and models/) entries to the ALLFED shared store.
#
# Usage:  scripts/link_shared.sh            make or refresh the links, then report
#         scripts/link_shared.sh --check    report only, change nothing
#
# Reads data/shared.tsv (tab-separated, '#' starts a comment):
#     <link path, relative to the repo>  <path inside the store>  <where to get it without the store>
#
# The store is the folder that holds shared-data/ and shared-models/. Default: the folder that
# contains this repository (e.g. ~/ALLFED). Override with ALLFED_STORE=/path/to/store.
# Links are relative (ln -r), so moving the whole tree keeps them valid. They must be git-ignored:
# the script warns about any link path that git would track.
# Exit status: 0 all present, 1 something missing, 2 usage or manifest error.
set -euo pipefail
check=0; [[ "${1:-}" == "--check" ]] && check=1
[[ $# -gt 1 || ( $# -eq 1 && $check -eq 0 ) ]] && { echo "usage: $0 [--check]" >&2; exit 2; }
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
store=${ALLFED_STORE:-$(dirname "$repo")}
manifest="$repo/data/shared.tsv"
[[ -f "$manifest" ]] || { echo "no $manifest" >&2; exit 2; }
missing=0
while IFS=$'\t' read -r link target source || [[ -n "${link:-}" ]]; do
  [[ -z "${link// }" || "$link" == \#* ]] && continue
  [[ -z "${target:-}" ]] && { echo "bad line in data/shared.tsv: $link" >&2; exit 2; }
  dst="$store/$target"; lnk="$repo/$link"
  if git -C "$repo" ls-files --error-unmatch -- "$link" >/dev/null 2>&1 \
     || ! git -C "$repo" check-ignore -q -- "$link" 2>/dev/null; then
    [[ -d "$repo/.git" || -f "$repo/.git" ]] && echo "WARNING $link is not git-ignored (add it to .gitignore)"
  fi
  if [[ -e "$lnk" && ! -L "$lnk" ]]; then
    echo "local   $link (real file or folder, not a link: left alone)"; continue
  fi
  if [[ ! -e "$dst" ]]; then
    echo "MISSING $link  <- $target   get it from: ${source:-see the Data section of README.md}"
    missing=1; continue
  fi
  if (( check )); then
    [[ -L "$lnk" && "$(readlink -f "$lnk")" == "$(readlink -f "$dst")" ]] \
      && echo "ok      $link" || { echo "UNLINKED $link (run without --check)"; missing=1; }
  else
    mkdir -p "$(dirname "$lnk")"
    ln -sfnr "$dst" "$lnk"
    echo "linked  $link -> $target"
  fi
done < "$manifest"
exit $missing
