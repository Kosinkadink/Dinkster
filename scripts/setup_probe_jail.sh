#!/usr/bin/env bash
set -euo pipefail

apt_with_retry() {
    local attempt
    for attempt in 1 2; do
        if sudo timeout 180 apt-get \
            -o Acquire::Retries=3 \
            -o Acquire::http::Timeout=30 \
            -o Acquire::https::Timeout=30 "$@"; then
            return 0
        fi
        echo "apt-get $1 failed or exceeded 180 seconds (attempt $attempt of 2)" >&2
    done
    return 1
}

apt_with_retry update
apt_with_retry install --yes --no-install-recommends bubblewrap apparmor-profiles

if [[ -f /proc/sys/kernel/apparmor_restrict_unprivileged_userns ]] &&
   [[ "$(cat /proc/sys/kernel/apparmor_restrict_unprivileged_userns)" = 1 ]]; then
    sudo apparmor_parser -r /usr/share/apparmor/extra-profiles/bwrap-userns-restrict
fi

bwrap --unshare-user --unshare-pid --unshare-net \
    --ro-bind / / --proc /proc --dev /dev /usr/bin/true
