#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install --yes --no-install-recommends bubblewrap apparmor-profiles

if [[ -f /proc/sys/kernel/apparmor_restrict_unprivileged_userns ]] &&
   [[ "$(cat /proc/sys/kernel/apparmor_restrict_unprivileged_userns)" = 1 ]]; then
    sudo apparmor_parser -r /usr/share/apparmor/extra-profiles/bwrap-userns-restrict
fi

bwrap --unshare-user --unshare-pid --unshare-net \
    --ro-bind / / --proc /proc --dev /dev /usr/bin/true
