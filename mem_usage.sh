#!/usr/bin/env bash

# mem_usage.sh
#
# Pretty Linux RAM / swap / CPU / Celery process inspector.
# No sudo required.
#
# Usage:
#   ./mem_usage.sh
#   ./mem_usage.sh 40
#
# Improvements over the original:
#   - Detects Celery beat/flower in addition to workers.
#   - Fills "-" cells with a short process name / role for non-Celery processes.
#   - Dynamic column widths based on actual content (ANSI/emoji safe).
#   - Per-worker and per-queue RAM summaries.
#   - Cleaner alignment and colour handling.
#
# Linux only.

set -u

LIMIT="${1:-25}"
SAMPLE_SECONDS="${SAMPLE_SECONDS:-1}"
MAX_WORKER_WIDTH=24
MAX_QUEUE_WIDTH=20

# ─────────────────────────────────────────────────────────────
# Colours
# ─────────────────────────────────────────────────────────────

if [[ -t 1 ]]; then
    RESET=$'\e[0m'
    BOLD=$'\e[1m'
    DIM=$'\e[2m'
    RED=$'\e[31m'
    GREEN=$'\e[32m'
    YELLOW=$'\e[33m'
    BLUE=$'\e[34m'
    MAGENTA=$'\e[35m'
    CYAN=$'\e[36m'
    WHITE=$'\e[97m'
else
    RESET=""
    BOLD=""
    DIM=""
    RED=""
    GREEN=""
    YELLOW=""
    BLUE=""
    MAGENTA=""
    CYAN=""
    WHITE=""
fi

if ! [[ "$LIMIT" =~ ^[0-9]+$ ]] || (( LIMIT < 1 )); then
    echo "Usage: $0 [number-of-processes]"
    exit 1
fi

# ─────────────────────────────────────────────────────────────
# Generic helpers
# ─────────────────────────────────────────────────────────────

human_kb() {
    local kb="${1:-0}"

    awk -v kb="$kb" '
        BEGIN {
            if (kb >= 1048576)
                printf "%.2f GiB", kb / 1048576
            else if (kb >= 1024)
                printf "%.1f MiB", kb / 1024
            else
                printf "%d KiB", kb
        }
    '
}

human_seconds() {
    local s="${1:-0}"

    if (( s < 60 )); then
        printf "%ds" "$s"
    elif (( s < 3600 )); then
        printf "%dm" "$((s / 60))"
    elif (( s < 86400 )); then
        printf "%dh%02dm" \
            "$((s / 3600))" \
            "$(((s % 3600) / 60))"
    else
        printf "%dd%02dh" \
            "$((s / 86400))" \
            "$(((s % 86400) / 3600))"
    fi
}

state_description() {
    case "$1" in
        R) printf "RUN" ;;
        S) printf "SLEEP" ;;
        D) printf "IOWAIT" ;;
        Z) printf "ZOMBIE" ;;
        T) printf "STOP" ;;
        t) printf "TRACE" ;;
        X|x) printf "DEAD" ;;
        I) printf "IDLE" ;;
        *) printf "%s" "$1" ;;
    esac
}

get_raw_cmdline() {
    local pid="$1"

    [[ -r "/proc/$pid/cmdline" ]] || return 1

    tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null |
        sed 's/[[:space:]]*$//'
}

read_proc_stat() {
    local pid="$1"
    local statfile="/proc/$pid/stat"
    local line rest

    [[ -r "$statfile" ]] || return 1

    IFS= read -r line < "$statfile" 2>/dev/null || return 1

    # Strip "PID (command)" prefix; everything remaining starts at field 3.
    rest="${line##*) }"

    read -r \
        p_state \
        p_ppid \
        _ _ _ _ _ _ _ \
        p_utime \
        p_stime \
        _ _ _ _ _ _ \
        p_starttime \
        _ <<< "$rest"

    return 0
}

visible_len() {
    local s="$1"
    printf '%s' "$s" | sed -E 's/\x1b\[[0-9;]*m//g' | wc -c
}

repeat_char() {
    local char="$1"
    local count="$2"
    printf '%*s' "$count" '' | tr ' ' "$char"
}

print_cell() {
    local value="$1"
    local width="$2"
    local color="${3:-}"
    local len pad

    len=$(visible_len "$value")
    pad=$((width - len))
    (( pad < 0 )) && pad=0

    if [[ -n "$color" ]]; then
        printf '%s%s%s%*s' "$color" "$value" "$RESET" "$pad" ""
    else
        printf '%s%*s' "$value" "$pad" ""
    fi
    printf ' '
}

# ─────────────────────────────────────────────────────────────
# Celery helpers
# ─────────────────────────────────────────────────────────────

is_celery_command() {
    local cmd="$1"

    [[ "$cmd" =~ (^|[[:space:]])([^[:space:]]*/)?celery([[:space:]]|$) ]] || return 1
    [[ "$cmd" =~ (^|[[:space:]])(worker|beat|flower)([[:space:]]|$) ]]
}

parse_celery_subcommand() {
    local cmd="$1"

    if [[ "$cmd" =~ (^|[[:space:]])flower([[:space:]]|$) ]]; then
        printf 'flower'
    elif [[ "$cmd" =~ (^|[[:space:]])beat([[:space:]]|$) ]]; then
        printf 'beat'
    elif [[ "$cmd" =~ (^|[[:space:]])worker([[:space:]]|$) ]]; then
        printf 'worker'
    fi
}

parse_celery_worker_name() {
    local cmd="$1"
    local value=""

    if [[ "$cmd" =~ (^|[[:space:]])-n[[:space:]]+([^[:space:]]+) ]]; then
        value="${BASH_REMATCH[2]}"
    elif [[ "$cmd" =~ (^|[[:space:]])-n=([^[:space:]]+) ]]; then
        value="${BASH_REMATCH[2]}"
    elif [[ "$cmd" =~ (^|[[:space:]])--hostname[[:space:]]+([^[:space:]]+) ]]; then
        value="${BASH_REMATCH[2]}"
    elif [[ "$cmd" =~ (^|[[:space:]])--hostname=([^[:space:]]+) ]]; then
        value="${BASH_REMATCH[2]}"
    fi

    # worker@hostname -> worker
    value="${value%%@*}"
    printf '%s' "$value"
}

parse_celery_queue() {
    local cmd="$1"
    local value=""

    if [[ "$cmd" =~ (^|[[:space:]])-Q[[:space:]]+([^[:space:]]+) ]]; then
        value="${BASH_REMATCH[2]}"
    elif [[ "$cmd" =~ (^|[[:space:]])-Q=([^[:space:]]+) ]]; then
        value="${BASH_REMATCH[2]}"
    elif [[ "$cmd" =~ (^|[[:space:]])--queues[[:space:]]+([^[:space:]]+) ]]; then
        value="${BASH_REMATCH[2]}"
    elif [[ "$cmd" =~ (^|[[:space:]])--queues=([^[:space:]]+) ]]; then
        value="${BASH_REMATCH[2]}"
    fi

    printf '%s' "$value"
}

parse_celery_concurrency() {
    local cmd="$1"
    local value=""

    if [[ "$cmd" =~ (^|[[:space:]])-c[[:space:]]+([0-9]+) ]]; then
        value="${BASH_REMATCH[2]}"
    elif [[ "$cmd" =~ (^|[[:space:]])-c=([0-9]+) ]]; then
        value="${BASH_REMATCH[2]}"
    elif [[ "$cmd" =~ (^|[[:space:]])--concurrency[[:space:]]+([0-9]+) ]]; then
        value="${BASH_REMATCH[2]}"
    elif [[ "$cmd" =~ (^|[[:space:]])--concurrency=([0-9]+) ]]; then
        value="${BASH_REMATCH[2]}"
    fi

    printf '%s' "$value"
}

parse_celery_pool() {
    local cmd="$1"
    local value=""

    if [[ "$cmd" =~ (^|[[:space:]])-P[[:space:]]+([^[:space:]]+) ]]; then
        value="${BASH_REMATCH[2]}"
    elif [[ "$cmd" =~ (^|[[:space:]])-P=([^[:space:]]+) ]]; then
        value="${BASH_REMATCH[2]}"
    elif [[ "$cmd" =~ (^|[[:space:]])--pool[[:space:]]+([^[:space:]]+) ]]; then
        value="${BASH_REMATCH[2]}"
    elif [[ "$cmd" =~ (^|[[:space:]])--pool=([^[:space:]]+) ]]; then
        value="${BASH_REMATCH[2]}"
    fi

    printf '%s' "$value"
}

# Walk upward through PPIDs until we find the Celery master.
find_celery_master_pid() {
    local current="$1"
    local depth=0
    local max_depth=20

    while [[ "$current" =~ ^[0-9]+$ ]] &&
          (( current > 1 && depth < max_depth )); do

        local cmd=""
        local line=""
        local rest=""
        local parent=""

        cmd="$(get_raw_cmdline "$current" 2>/dev/null || true)"

        if [[ -n "$cmd" ]] && is_celery_command "$cmd" &&
           [[ "$(parse_celery_subcommand "$cmd")" == "worker" ]]; then
            printf '%s' "$current"
            return 0
        fi

        [[ -r "/proc/$current/stat" ]] || break

        IFS= read -r line < "/proc/$current/stat" 2>/dev/null || break

        rest="${line##*) }"
        read -r _ parent _ <<< "$rest"

        [[ "$parent" =~ ^[0-9]+$ ]] || break
        (( parent == current )) && break

        current="$parent"
        ((depth++))
    done

    return 1
}

short_process_name() {
    local cmd="$1"
    local IFS=' '
    local tokens=($cmd)
    local first="${tokens[0]:-unknown}"

    first="${first##*/}"
    first="${first%%:*}"
    first="${first#\[}"     # kernel threads like [kthreadd]
    first="${first%\]}"
    first="${first#"${first%%[![:space:]]*}"}"   # trim leading whitespace

    [[ -n "$first" ]] || first="unknown"

    # Keep the display name compact so the table stays aligned.
    if (( ${#first} > 20 )); then
        first="${first:0:20}"
    fi

    printf '%s' "$first"
}

# Returns: type|worker|queue|concurrency|pool
get_process_metadata() {
    local pid="$1"
    local cmd=""
    local sub=""
    local master_pid=""
    local master_cmd=""
    local worker="" queue="" conc="" pool="" type="OTHER"

    cmd="$(get_raw_cmdline "$pid" 2>/dev/null || true)"
    [[ -n "$cmd" ]] || cmd="unknown"

    if is_celery_command "$cmd"; then
        sub="$(parse_celery_subcommand "$cmd")"

        case "$sub" in
            flower)
                worker="$(parse_celery_worker_name "$cmd")"
                [[ -n "$worker" ]] || worker="flower"
                printf 'FLOWER|%s|-|-|-\n' "$worker"
                return
                ;;
            beat)
                worker="$(parse_celery_worker_name "$cmd")"
                [[ -n "$worker" ]] || worker="beat"
                printf 'BEAT|%s|-|-|-\n' "$worker"
                return
                ;;
        esac
    fi

    # Try to resolve a Celery master above this process. This catches
    # billiard/prefork children whose own cmdline is just
    #   python -c from billiard.spawn import spawn_main ...
    master_pid="$(find_celery_master_pid "$pid" 2>/dev/null || true)"

    if [[ -n "$master_pid" ]]; then
        master_cmd="$(get_raw_cmdline "$master_pid" 2>/dev/null || true)"
        worker="$(parse_celery_worker_name "$master_cmd")"
        queue="$(parse_celery_queue "$master_cmd")"
        conc="$(parse_celery_concurrency "$master_cmd")"
        pool="$(parse_celery_pool "$master_cmd")"

        if [[ "$pid" == "$master_pid" ]]; then
            type="MASTER"
        else
            type="CHILD"
        fi
    elif is_celery_command "$cmd"; then
        # A Celery worker process whose master could not be resolved.
        worker="$(parse_celery_worker_name "$cmd")"
        queue="$(parse_celery_queue "$cmd")"
        conc="$(parse_celery_concurrency "$cmd")"
        pool="$(parse_celery_pool "$cmd")"
        type="WORKER"
    else
        worker="$(short_process_name "$cmd")"
        printf 'OTHER|%s|-|-|-\n' "$worker"
        return
    fi

    [[ -n "$worker" ]] || worker="default"
    [[ -n "$queue" ]] || queue="default"
    [[ -n "$conc" ]] || conc="?"
    [[ -n "$pool" ]] || pool="default"

    printf '%s|%s|%s|%s|%s\n' "$type" "$worker" "$queue" "$conc" "$pool"
}

# ─────────────────────────────────────────────────────────────
# System info
# ─────────────────────────────────────────────────────────────

CLK_TCK="$(
    getconf CLK_TCK 2>/dev/null || echo 100
)"

CPU_COUNT="$(
    getconf _NPROCESSORS_ONLN 2>/dev/null ||
    nproc 2>/dev/null ||
    echo 1
)"

UPTIME_SECONDS="$(
    awk '{printf "%d", $1}' /proc/uptime
)"

mem_total="$(
    awk '/^MemTotal:/ {print $2}' /proc/meminfo
)"

mem_avail="$(
    awk '/^MemAvailable:/ {print $2}' /proc/meminfo
)"

swap_total="$(
    awk '/^SwapTotal:/ {print $2}' /proc/meminfo
)"

swap_free="$(
    awk '/^SwapFree:/ {print $2}' /proc/meminfo
)"

mem_used=$((mem_total - mem_avail))
swap_used=$((swap_total - swap_free))

# ─────────────────────────────────────────────────────────────
# CPU sample #1
# ─────────────────────────────────────────────────────────────

declare -A CPU_BEFORE

for procdir in /proc/[0-9]*; do
    pid="${procdir##*/}"

    if read_proc_stat "$pid"; then
        CPU_BEFORE["$pid"]=$((p_utime + p_stime))
    fi
done

sleep "$SAMPLE_SECONDS"

# ─────────────────────────────────────────────────────────────
# Collect processes
# ─────────────────────────────────────────────────────────────

declare -a rows=()

total_rss=0
total_swap=0

active_count=0
idle_count=0
zombie_count=0
adopted_count=0

for procdir in /proc/[0-9]*; do

    [[ -r "$procdir/status" ]] || continue

    pid="${procdir##*/}"

    name=""
    rss=0
    swap=0
    uid=""
    threads=0

    while IFS=$': \t' read -r key value _; do
        case "$key" in
            Name)
                name="$value"
                ;;
            VmRSS)
                rss="${value%% *}"
                ;;
            VmSwap)
                swap="${value%% *}"
                ;;
            Uid)
                uid="$value"
                ;;
            Threads)
                threads="$value"
                ;;
        esac
    done < "$procdir/status" 2>/dev/null

    [[ "$rss" =~ ^[0-9]+$ ]] || rss=0
    [[ "$swap" =~ ^[0-9]+$ ]] || swap=0
    [[ "$threads" =~ ^[0-9]+$ ]] || threads=0

    read_proc_stat "$pid" || continue

    state="$p_state"
    ppid="$p_ppid"
    cpu_after=$((p_utime + p_stime))
    start_ticks="$p_starttime"

    # Ignore zero-memory entries.
    (( rss == 0 && swap == 0 )) && continue

    # User
    user="$(
        getent passwd "$uid" 2>/dev/null |
        cut -d: -f1
    )"
    [[ -n "$user" ]] || user="$uid"

    # Full RAW command
    raw_cmd="$(get_raw_cmdline "$pid" 2>/dev/null || true)"
    [[ -n "$raw_cmd" ]] || raw_cmd="$name"

    # Celery / process metadata
    IFS='|' read -r \
        celery_type \
        celery_worker \
        celery_queue \
        celery_concurrency \
        celery_pool \
        <<< "$(get_process_metadata "$pid")"

    # Sanitised display command
    cmd="$(
        printf '%s' "$raw_cmd" |
            sed 's#/home/dockerhost/graphai/\.venv\.graphai/bin/##g' |
            sed -E 's#--broker=[^[:space:]]+#--broker=[REDACTED]#g'
    )"

    # CPU
    cpu_before="${CPU_BEFORE[$pid]:-$cpu_after}"
    delta_ticks=$((cpu_after - cpu_before))
    (( delta_ticks < 0 )) && delta_ticks=0

    cpu_percent="$(
        awk \
            -v ticks="$delta_ticks" \
            -v hz="$CLK_TCK" \
            -v sec="$SAMPLE_SECONDS" '
            BEGIN {
                if (hz <= 0 || sec <= 0)
                    printf "0.0"
                else
                    printf "%.1f", (ticks / hz / sec) * 100
            }
        '
    )"

    # Process age
    start_seconds=$((start_ticks / CLK_TCK))
    age_seconds=$((UPTIME_SECONDS - start_seconds))
    (( age_seconds < 0 )) && age_seconds=0
    age="$(human_seconds "$age_seconds")"

    # Adopted by PID 1
    adopted=0
    if (( pid != 1 && ppid == 1 )); then
        adopted=1
        ((adopted_count++))
    fi

    # Activity classification
    if [[ "$state" == "Z" ]]; then
        activity="ZOMBIE"
        ((zombie_count++))
    elif [[ "$state" == "D" ]]; then
        activity="IOWAIT"
        ((active_count++))
    elif awk -v c="$cpu_percent" 'BEGIN { exit !(c >= 50) }'; then
        activity="HOT"
        ((active_count++))
    elif awk -v c="$cpu_percent" 'BEGIN { exit !(c >= 5) }'; then
        activity="ACTIVE"
        ((active_count++))
    elif awk -v c="$cpu_percent" 'BEGIN { exit !(c >= 0.2) }'; then
        activity="BUSY"
        ((active_count++))
    elif [[ "$state" == "R" ]]; then
        activity="RUNNING"
        ((active_count++))
    elif (( adopted )); then
        activity="ADOPTED"
        ((idle_count++))
    else
        activity="QUIET"
        ((idle_count++))
    fi

    combined=$((rss + swap))

    (( total_rss += rss ))
    (( total_swap += swap ))

    rows+=(
        "$combined|$pid|$user|$rss|$swap|$cpu_percent|$state|$ppid|$age|$activity|$adopted|$celery_type|$celery_worker|$celery_queue|$celery_concurrency|$celery_pool|$cmd"
    )

done

# ─────────────────────────────────────────────────────────────
# First pass: format values, measure widths, build summaries
# ─────────────────────────────────────────────────────────────

declare -a a_pid=() a_user=() a_rss=() a_swap=() a_cpu=() a_state=() a_activity=()
declare -a a_ppid=() a_age=() a_type=() a_worker=() a_queue=() a_conc=() a_pool=() a_cmd=()
declare -A worker_rss=() worker_procs=() queue_rss=() queue_procs=()

w_pid=3 w_user=4 w_rss=3 w_swap=4 w_cpu=4 w_state=5 w_activity=8 w_ppid=4 w_age=3
w_type=4 w_worker=6 w_queue=5 w_conc=4 w_pool=4

row_count=0

while IFS='|' read -r \
    combined \
    pid \
    user \
    rss \
    swap \
    cpu \
    state \
    ppid \
    age \
    activity \
    adopted \
    celery_type \
    celery_worker \
    celery_queue \
    celery_concurrency \
    celery_pool \
    cmd

do
    ((row_count++))

    f_pid="$pid"
    f_user="$user"
    f_rss="$(human_kb "$rss")"
    f_swap="$(human_kb "$swap")"
    f_cpu="${cpu}%"
    f_state="$(state_description "$state")"
    f_activity="$activity"
    f_ppid="$ppid"
    f_age="$age"
    # Keep long names from blowing out the table.
    if (( ${#celery_worker} > MAX_WORKER_WIDTH )); then
        celery_worker="${celery_worker:0:MAX_WORKER_WIDTH}"
    fi
    if (( ${#celery_queue} > MAX_QUEUE_WIDTH )); then
        celery_queue="${celery_queue:0:MAX_QUEUE_WIDTH}"
    fi

    f_type="$celery_type"
    f_worker="$celery_worker"
    f_queue="$celery_queue"
    f_conc="$celery_concurrency"
    f_pool="$celery_pool"
    f_cmd="$cmd"

    a_pid+=("$f_pid")
    a_user+=("$f_user")
    a_rss+=("$f_rss")
    a_swap+=("$f_swap")
    a_cpu+=("$f_cpu")
    a_state+=("$f_state")
    a_activity+=("$f_activity")
    a_ppid+=("$f_ppid")
    a_age+=("$f_age")
    a_type+=("$f_type")
    a_worker+=("$f_worker")
    a_queue+=("$f_queue")
    a_conc+=("$f_conc")
    a_pool+=("$f_pool")
    a_cmd+=("$f_cmd")

    # Update widths
    (( $(visible_len "$f_pid") > w_pid )) && w_pid=$(visible_len "$f_pid")
    (( $(visible_len "$f_user") > w_user )) && w_user=$(visible_len "$f_user")
    (( $(visible_len "$f_rss") > w_rss )) && w_rss=$(visible_len "$f_rss")
    (( $(visible_len "$f_swap") > w_swap )) && w_swap=$(visible_len "$f_swap")
    (( $(visible_len "$f_cpu") > w_cpu )) && w_cpu=$(visible_len "$f_cpu")
    (( $(visible_len "$f_state") > w_state )) && w_state=$(visible_len "$f_state")
    (( $(visible_len "$f_activity") > w_activity )) && w_activity=$(visible_len "$f_activity")
    (( $(visible_len "$f_ppid") > w_ppid )) && w_ppid=$(visible_len "$f_ppid")
    (( $(visible_len "$f_age") > w_age )) && w_age=$(visible_len "$f_age")
    (( $(visible_len "$f_type") > w_type )) && w_type=$(visible_len "$f_type")
    (( $(visible_len "$f_worker") > w_worker )) && w_worker=$(visible_len "$f_worker")
    (( $(visible_len "$f_queue") > w_queue )) && w_queue=$(visible_len "$f_queue")
    (( $(visible_len "$f_conc") > w_conc )) && w_conc=$(visible_len "$f_conc")
    (( $(visible_len "$f_pool") > w_pool )) && w_pool=$(visible_len "$f_pool")

    # Summaries for Celery workers
    if [[ "$celery_type" == "MASTER" || "$celery_type" == "CHILD" ]]; then
        worker_rss["$celery_worker"]=$(( ${worker_rss["$celery_worker"]:-0} + rss ))
        worker_procs["$celery_worker"]=$(( ${worker_procs["$celery_worker"]:-0} + 1 ))
        queue_rss["$celery_queue"]=$(( ${queue_rss["$celery_queue"]:-0} + rss ))
        queue_procs["$celery_queue"]=$(( ${queue_procs["$celery_queue"]:-0} + 1 ))
    fi

done < <(
    printf '%s\n' "${rows[@]}" |
        sort -t'|' -k1,1nr
)

# Ensure widths are at least header widths
(( $(visible_len "PID") > w_pid )) && w_pid=$(visible_len "PID")
(( $(visible_len "USER") > w_user )) && w_user=$(visible_len "USER")
(( $(visible_len "RAM") > w_rss )) && w_rss=$(visible_len "RAM")
(( $(visible_len "SWAP") > w_swap )) && w_swap=$(visible_len "SWAP")
(( $(visible_len "CPU%") > w_cpu )) && w_cpu=$(visible_len "CPU%")
(( $(visible_len "STATE") > w_state )) && w_state=$(visible_len "STATE")
(( $(visible_len "ACTIVITY") > w_activity )) && w_activity=$(visible_len "ACTIVITY")
(( $(visible_len "PPID") > w_ppid )) && w_ppid=$(visible_len "PPID")
(( $(visible_len "AGE") > w_age )) && w_age=$(visible_len "AGE")
(( $(visible_len "TYPE") > w_type )) && w_type=$(visible_len "TYPE")
(( $(visible_len "WORKER") > w_worker )) && w_worker=$(visible_len "WORKER")
(( $(visible_len "QUEUE") > w_queue )) && w_queue=$(visible_len "QUEUE")
(( $(visible_len "CONC") > w_conc )) && w_conc=$(visible_len "CONC")
(( $(visible_len "POOL") > w_pool )) && w_pool=$(visible_len "POOL")

# Cap descriptive columns so the table does not get absurdly wide.
(( w_worker > MAX_WORKER_WIDTH )) && w_worker=$MAX_WORKER_WIDTH
(( w_queue > MAX_QUEUE_WIDTH )) && w_queue=$MAX_QUEUE_WIDTH

# Terminal width and command column width
TERM_WIDTH="$(tput cols 2>/dev/null || echo 120)"
(( TERM_WIDTH < 80 )) && TERM_WIDTH=80

total_fixed_width=$((
    w_pid + w_user + w_rss + w_swap + w_cpu + w_state + w_activity +
    w_ppid + w_age + w_type + w_worker + w_queue + w_conc + w_pool + 14
))
w_cmd=$((TERM_WIDTH - total_fixed_width))
(( w_cmd < 20 )) && w_cmd=20

# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────

clear 2>/dev/null || true

printf "%s%s🧠  PROCESS MEMORY / SWAP / CPU / CELERY INSPECTOR%s\n" \
    "$BOLD" "$CYAN" "$RESET"

printf "%s%s%s\n" "$DIM" "$(repeat_char '-' 80)" "$RESET"

printf "💻 RAM     %s%s%s / %s\n" \
    "$GREEN" "$(human_kb "$mem_used")" "$RESET" "$(human_kb "$mem_total")"

if (( swap_total > 0 )); then
    printf "💾 SWAP    %s%s%s / %s\n" \
        "$MAGENTA" "$(human_kb "$swap_used")" "$RESET" "$(human_kb "$swap_total")"
else
    printf "💾 SWAP    %sdisabled%s\n" "$DIM" "$RESET"
fi

printf "🧮 CPUs    %s%s logical CPUs%s\n" "$CYAN" "$CPU_COUNT" "$RESET"
printf "⏱️  Sample  %ss\n" "$SAMPLE_SECONDS"
printf "\n"
printf "%sTop %d processes by RAM + swap%s\n\n" "$BOLD" "$LIMIT" "$RESET"

# ─────────────────────────────────────────────────────────────
# Table
# ─────────────────────────────────────────────────────────────

print_cell "PID" "$w_pid" "$BOLD"
print_cell "USER" "$w_user" "$BOLD"
print_cell "RAM" "$w_rss" "$BOLD"
print_cell "SWAP" "$w_swap" "$BOLD"
print_cell "CPU%" "$w_cpu" "$BOLD"
print_cell "STATE" "$w_state" "$BOLD"
print_cell "ACTIVITY" "$w_activity" "$BOLD"
print_cell "PPID" "$w_ppid" "$BOLD"
print_cell "AGE" "$w_age" "$BOLD"
print_cell "TYPE" "$w_type" "$BOLD"
print_cell "WORKER" "$w_worker" "$BOLD"
print_cell "QUEUE" "$w_queue" "$BOLD"
print_cell "CONC" "$w_conc" "$BOLD"
print_cell "POOL" "$w_pool" "$BOLD"
printf "%s%s%s\n" "$BOLD" "COMMAND" "$RESET"

printf "%s%s%s\n" "$DIM" "$(repeat_char '-' "$((total_fixed_width + w_cmd))")" "$RESET"

# ─────────────────────────────────────────────────────────────
# Rows
# ─────────────────────────────────────────────────────────────

for ((i = 0; i < row_count && i < LIMIT; i++)); do

    pid="${a_pid[$i]}"
    user="${a_user[$i]}"
    rss="${a_rss[$i]}"
    swap="${a_swap[$i]}"
    cpu="${a_cpu[$i]}"
    state="${a_state[$i]}"
    activity="${a_activity[$i]}"
    ppid="${a_ppid[$i]}"
    age="${a_age[$i]}"
    ptype="${a_type[$i]}"
    worker="${a_worker[$i]}"
    queue="${a_queue[$i]}"
    conc="${a_conc[$i]}"
    pool="${a_pool[$i]}"
    cmd="${a_cmd[$i]}"

    # Truncate command to fit column
    if (( ${#cmd} > w_cmd )); then
        cmd="${cmd:0:w_cmd}"
    fi

    # RAM colour
    rss_kb_raw="${rss% *}"  # not needed for colour; use value from formatted
    if [[ "$rss" =~ GiB ]]; then
        ram_colour="$RED"
    elif [[ "$rss" =~ MiB ]]; then
        # extract numeric part
        rss_num="${rss% MiB}"
        if awk -v n="$rss_num" 'BEGIN { exit !(n >= 1024) }'; then
            ram_colour="$YELLOW"
        else
            ram_colour="$GREEN"
        fi
    else
        ram_colour="$GREEN"
    fi

    # Swap colour
    if [[ "$swap" != "0 KiB" ]]; then
        if [[ "$swap" =~ GiB ]]; then
            swap_colour="$RED"
        elif [[ "$swap" =~ MiB ]]; then
            swap_num="${swap% MiB}"
            if awk -v n="$swap_num" 'BEGIN { exit !(n >= 256) }'; then
                swap_colour="$YELLOW"
            else
                swap_colour="$MAGENTA"
            fi
        else
            swap_colour="$MAGENTA"
        fi
    else
        swap_colour="$DIM"
    fi

    # CPU colour
    cpu_num="${cpu%\%}"
    if awk -v c="$cpu_num" 'BEGIN { exit !(c >= 100) }'; then
        cpu_colour="$RED"
    elif awk -v c="$cpu_num" 'BEGIN { exit !(c >= 25) }'; then
        cpu_colour="$YELLOW"
    elif awk -v c="$cpu_num" 'BEGIN { exit !(c >= 1) }'; then
        cpu_colour="$GREEN"
    else
        cpu_colour="$DIM"
    fi

    # State colour
    case "${a_state[$i]}" in
        RUN)   state_colour="$GREEN" ;;
        IOWAIT) state_colour="$YELLOW" ;;
        ZOMBIE) state_colour="$RED" ;;
        STOP|TRACE) state_colour="$MAGENTA" ;;
        *)     state_colour="$DIM" ;;
    esac

    # Activity colour
    case "$activity" in
        HOT)     activity_colour="$RED" ;;
        ACTIVE)  activity_colour="$YELLOW" ;;
        BUSY|RUNNING) activity_colour="$GREEN" ;;
        IOWAIT)  activity_colour="$YELLOW" ;;
        ZOMBIE)  activity_colour="$RED" ;;
        ADOPTED) activity_colour="$YELLOW" ;;
        *)       activity_colour="$DIM" ;;
    esac

    # Type colour
    case "$ptype" in
        MASTER)  type_colour="$CYAN" ;;
        CHILD)   type_colour="$MAGENTA" ;;
        BEAT)    type_colour="$YELLOW" ;;
        FLOWER)  type_colour="$BLUE" ;;
        *)       type_colour="$DIM" ;;
    esac

    print_cell "$pid" "$w_pid" ""
    print_cell "$user" "$w_user" ""
    print_cell "$rss" "$w_rss" "$ram_colour"
    print_cell "$swap" "$w_swap" "$swap_colour"
    print_cell "$cpu" "$w_cpu" "$cpu_colour"
    print_cell "$state" "$w_state" "$state_colour"
    print_cell "$activity" "$w_activity" "$activity_colour"
    print_cell "$ppid" "$w_ppid" ""
    print_cell "$age" "$w_age" ""
    print_cell "$ptype" "$w_type" "$type_colour"
    print_cell "$worker" "$w_worker" "$type_colour"
    print_cell "$queue" "$w_queue" ""
    print_cell "$conc" "$w_conc" ""
    print_cell "$pool" "$w_pool" ""
    printf "%s\n" "$cmd"

done

# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────

printf "\n"
printf "%s%s%s\n" "$DIM" "$(repeat_char '-' "$((total_fixed_width + w_cmd))")" "$RESET"

printf "📊 Visible process totals: RAM %s%s%s   SWAP %s%s%s\n" \
    "$GREEN" "$(human_kb "$total_rss")" "$RESET" \
    "$MAGENTA" "$(human_kb "$total_swap")" "$RESET"

printf "⚡ Active: %s%d%s   💤 Quiet: %s%d%s   🧟 Zombies: %s%d%s   👶 Adopted: %s%d%s\n" \
    "$GREEN" "$active_count" "$RESET" \
    "$DIM" "$idle_count" "$RESET" \
    "$RED" "$zombie_count" "$RESET" \
    "$YELLOW" "$adopted_count" "$RESET"

printf "\n"

# ─────────────────────────────────────────────────────────────
# Per-worker summary
# ─────────────────────────────────────────────────────────────

if (( ${#worker_rss[@]} > 0 )); then
    printf "%sTop workers by RAM%s\n" "$BOLD" "$RESET"
    printf "  %-*s %6s %12s\n" "$w_worker" "WORKER" "PROCS" "RAM"

    summary_rows=()
    for w in "${!worker_rss[@]}"; do
        summary_rows+=("${worker_rss[$w]}|${worker_procs[$w]}|$w")
    done

    while IFS='|' read -r rss_kb procs w; do
        printf "  %-*s %6s %12s\n" \
            "$w_worker" "$w" "$procs" "$(human_kb "$rss_kb")"
    done < <(
        printf '%s\n' "${summary_rows[@]}" | sort -t'|' -k1,1nr
    )

    printf "\n"
fi

# ─────────────────────────────────────────────────────────────
# Per-queue summary
# ─────────────────────────────────────────────────────────────

if (( ${#queue_rss[@]} > 0 )); then
    printf "%sTop queues by RAM%s\n" "$BOLD" "$RESET"
    printf "  %-*s %6s %12s\n" "$w_queue" "QUEUE" "PROCS" "RAM"

    summary_rows=()
    for q in "${!queue_rss[@]}"; do
        summary_rows+=("${queue_rss[$q]}|${queue_procs[$q]}|$q")
    done

    while IFS='|' read -r rss_kb procs q; do
        printf "  %-*s %6s %12s\n" \
            "$w_queue" "$q" "$procs" "$(human_kb "$rss_kb")"
    done < <(
        printf '%s\n' "${summary_rows[@]}" | sort -t'|' -k1,1nr
    )

    printf "\n"
fi

# ─────────────────────────────────────────────────────────────
# Legend
# ─────────────────────────────────────────────────────────────

printf "%sCelery / process types:%s\n" "$BOLD" "$RESET"
printf "  %sMASTER%s  = Celery worker master process\n" "$CYAN" "$RESET"
printf "  %sCHILD%s   = Billiard/prefork child of a worker\n" "$MAGENTA" "$RESET"
printf "  %sBEAT%s    = Celery beat scheduler\n" "$YELLOW" "$RESET"
printf "  %sFLOWER%s  = Celery Flower monitoring UI\n" "$BLUE" "$RESET"
printf "  %sOTHER%s   = Non-Celery process (short name in WORKER column)\n" "$DIM" "$RESET"
printf "  WORKER  = -n / --hostname (or short process name)\n"
printf "  QUEUE   = -Q / --queues\n"
printf "  CONC    = -c / --concurrency\n"
printf "  POOL    = -P / --pool\n"

printf "\n"

printf "%sActivity legend:%s\n" "$BOLD" "$RESET"
printf "  🔥 HOT       ≥ 50%% CPU during sample\n"
printf "  ⚡ ACTIVE     ≥ 5%% CPU during sample\n"
printf "  🟢 BUSY       ≥ 0.2%% CPU during sample / actively running\n"
printf "  💤 QUIET      negligible CPU during sample\n"
printf "  💿 IO WAIT    blocked in uninterruptible I/O\n"
printf "  🧟 ZOMBIE     exited, parent has not reaped it\n"
printf "  👶 ADOPTED    PPID=1\n"

printf "\n"

printf "%sCPU%% sampled over %ss. 100%% ≈ one fully occupied logical CPU.%s\n" \
    "$DIM" "$SAMPLE_SECONDS" "$RESET"

printf "%sBroker URLs are redacted from displayed command lines.%s\n" \
    "$DIM" "$RESET"

printf "%sNo sudo required; /proc permissions may hide some other-user details.%s\n" \
    "$DIM" "$RESET"
