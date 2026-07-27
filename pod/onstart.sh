#!/bin/bash
# Canonical launcher for the motion server on the pod. Lives on the network volume at
# /workspace/onstart.sh so it survives pod termination. Idempotent — safe to run on every
# boot or repeatedly by hand; exits cleanly if the server is already up.
if pgrep -f "[m]otion_server.py" >/dev/null; then
    echo "motion server already running"
    exit 0
fi
cd /workspace/app || exit 1
. /workspace/envs/mdm/bin/activate
nohup python -W ignore motion_server.py > /workspace/server.log 2>&1 &
echo "motion server launched, pid $!"
