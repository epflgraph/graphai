#!/usr/bin/env bash

printf "\n🤖 Graph AI\n\n"

docker compose ps --format '{{.Service}}|{{.Status}}|{{.Ports}}' |
while IFS='|' read -r service status ports; do
    case "$status" in
        *healthy*)
            icon="🟢"
            ;;
        Up*)
            icon="🟢"
            ;;
        *Restarting*|*unhealthy*)
            icon="🔴"
            ;;
        *Exited*|*Dead*)
            icon="🔴"
            ;;
        *)
            icon="🟡"
            ;;
    esac

    printf "%-3s %-16s %-32s %s\n" "$icon" "$service" "$status" "$ports"
done

echo
