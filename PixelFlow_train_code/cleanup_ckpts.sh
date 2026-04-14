#!/bin/bash
# Cleanup checkpoints: keep every 10000 steps + the latest one
# Runs every 1 hour until killed

DIR="/home/ruy45/MSFlow/PixelFlow_train_code/exp0001"

while true; do
    LATEST=$(ls -t "$DIR"/model_*.pt 2>/dev/null | head -1)

    for f in "$DIR"/model_*.pt; do
        [ -f "$f" ] || continue
        basename=$(basename "$f")
        [[ "$basename" == "model_final.pt" ]] && continue

        step=$(echo "$basename" | grep -oP '\d+')
        [ -z "$step" ] && continue

        # Keep if divisible by 10000
        if (( step % 100000 == 0 )); then
            continue
        fi

        # Keep if it's the latest file
        if [ "$f" = "$LATEST" ]; then
            continue
        fi

        rm -f "$f"
        echo "$(date '+%H:%M:%S') Deleted $basename"
    done

    sleep 3600
done
