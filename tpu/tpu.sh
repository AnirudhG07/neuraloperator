#!/usr/bin/env bash
# Interactive helper for a single-chip Cloud TPU research VM — neuraloperator/tpu edition.
# fzf-driven menu; the preview pane shows the exact gcloud command, and nothing runs
# until you confirm.  Adapted from pallas_tpu/tpu.sh for the FFT/FNO bench:
#   • uploads THIS repo's tpu/ tree to ~/nop on the VM  (so `import tpu.experimental.bench.bench_jax`)
#   • borrows the existing pallas_tpu uv venv (it already has jax[tpu]) via PYTHONPATH=~/nop
#   • `run` default = the bench (compare + profile) on device
# xprof needs TensorFlow only to PARSE traces; the VM lacks it, so `profile()` just captures
# *.xplane.pb and you `copyback` + run `bench_jax.report()` on a box that has TF.
#
# Override any default via env, e.g.:  TPU_ZONE=us-central1-a ./tpu.sh
set -uo pipefail

PROJECT_DEFAULT="fno-tpu"
STATE_ZONE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.tpu_zone"
ZONE="${TPU_ZONE:-$( [ -f "$STATE_ZONE" ] && cat "$STATE_ZONE" || echo us-west1-c )}"
ZONE_POOL="${TPU_ZONE_POOL:-us-west1-c us-west4-a us-central1-a us-east1-c us-east5-b asia-southeast1-b asia-east1-a asia-east1-c asia-northeast1-b europe-west4-b}"
VM_NAME="${TPU_NAME:-tpu-aces-ani}"
ACCEL="${TPU_ACCEL:-v5litepod-1}"
VERSION="${TPU_VERSION:-v2-alpha-tpuv5-lite}"
SPOT="${TPU_SPOT:-1}"                       # 1 = create with --spot (cheaper, preemptible)
REPO_URL="${TPU_REPO_URL:-https://github.com/sqtian/PALLAS_TPU_KERNEL_MATMUL}"
ENV_DIR="${TPU_ENV_DIR:-pallas_tpu}"        # remote dir whose uv venv has jax[tpu] (borrowed)
REMOTE_DIR="${TPU_REMOTE_DIR:-nop}"         # remote dir the uploaded tpu/ tree lands in (~/nop/tpu)
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
LOCAL_DIR="$(dirname "$SELF")"              # this = the repo's tpu/ directory
# The bench invocation reused by `run` (fresh ~/nopenv venv, ~/nop on the path).
BENCH_DEFAULT="export PATH=\$HOME/.local/bin:\$PATH; PYTHONPATH=\$HOME/$REMOTE_DIR ~/nopenv/bin/python -c \"import tpu.experimental.bench.bench_jax as bj; bj.compare(case='rfft_1D', Ns=(256,1024,4096,8192,16384))\""

# Find gcloud even if this shell predates the install.
if ! command -v gcloud >/dev/null 2>&1; then
  for c in "$HOME/google-cloud-sdk/bin" /usr/lib/google-cloud-sdk/bin; do
    [ -x "$c/gcloud" ] && PATH="$PATH:$c" && break
  done
fi
command -v gcloud >/dev/null 2>&1 || { echo "gcloud not found on PATH"; exit 1; }

PROJECT="${TPU_PROJECT:-$(gcloud config get-value project 2>/dev/null </dev/null)}"
[ "$PROJECT" = "(unset)" ] && PROJECT=""
[ -z "$PROJECT" ] && PROJECT="$PROJECT_DEFAULT"

ZONE_LIST="$ZONE"
for _z in $ZONE_POOL; do [ "$_z" != "$ZONE" ] && ZONE_LIST="$ZONE_LIST $_z"; done

bold=$'\e[1m'; dim=$'\e[2m'; grn=$'\e[32m'; ylw=$'\e[33m'; red=$'\e[31m'; cyn=$'\e[36m'; rst=$'\e[0m'

# ---------------------------------------------------------------- commands --
# cmd_for <id> prints the gcloud command for that action. Used both by the
# fzf preview pane and by the runner, so what you see is what executes.
cmd_for() {
  case "$1" in
    create)
      local c="gcloud compute tpus tpu-vm create $VM_NAME \\
  --zone=$ZONE \\
  --project=$PROJECT \\
  --accelerator-type=$ACCEL \\
  --version=$VERSION"
      [ "$SPOT" = 1 ] && c="$c \\
  --spot"
      echo "$c" ;;
    hunt)      local sp=""; [ "$SPOT" = 1 ] && sp=" \\
    --spot"
               echo "for z in $ZONE_LIST; do
  echo \"=== trying \$z\"
  gcloud compute tpus tpu-vm create $VM_NAME --zone=\$z --project=$PROJECT \\
    --accelerator-type=$ACCEL --version=$VERSION$sp \\
    && echo \$z > $STATE_ZONE \\
    && echo \"CREATED in \$z  (saved as the active zone)\" && break
done" ;;
    queue)     echo "gcloud compute tpus queued-resources create $VM_NAME-qr \\
  --node-id=$VM_NAME \\
  --zone=$ZONE \\
  --project=$PROJECT \\
  --accelerator-type=$ACCEL \\
  --runtime-version=$VERSION \\
  --spot \\
  --valid-until-duration=6h" ;;
    qstatus)   echo "gcloud compute tpus queued-resources list --zone=$ZONE --project=$PROJECT" ;;
    qdelete)   echo "gcloud compute tpus queued-resources delete $VM_NAME-qr --zone=$ZONE --project=$PROJECT --force --quiet" ;;
    start)     echo "gcloud compute tpus tpu-vm start $VM_NAME --zone=$ZONE --project=$PROJECT" ;;
    stop)      echo "gcloud compute tpus tpu-vm stop $VM_NAME --zone=$ZONE --project=$PROJECT" ;;
    delete)    echo "gcloud compute tpus tpu-vm delete $VM_NAME --zone=$ZONE --project=$PROJECT --quiet" ;;
    ssh)       echo "gcloud compute tpus tpu-vm ssh $VM_NAME --zone=$ZONE --project=$PROJECT" ;;
    run)       echo "gcloud compute tpus tpu-vm ssh $VM_NAME --zone=$ZONE --project=$PROJECT --worker=0 \\
  --command='$BENCH_DEFAULT'" ;;
    upload)    echo "tar czf /tmp/nop_tpu_upload.tar.gz -C $(dirname "$LOCAL_DIR") \\
  --exclude=traces --exclude=tpu_prof\\* --exclude='*.zip' --exclude=__pycache__ --exclude='*.png' --exclude=.venv \\
  $(basename "$LOCAL_DIR") && \\
gcloud compute tpus tpu-vm scp /tmp/nop_tpu_upload.tar.gz $VM_NAME:~/ \\
  --zone=$ZONE --project=$PROJECT --worker=0 && \\
gcloud compute tpus tpu-vm ssh $VM_NAME --zone=$ZONE --project=$PROJECT --worker=0 \\
  --command='rm -rf ~/$REMOTE_DIR && mkdir -p ~/$REMOTE_DIR && tar xzf ~/nop_tpu_upload.tar.gz -C ~/$REMOTE_DIR && ls ~/$REMOTE_DIR/tpu'" ;;
    bootstrap) echo "gcloud compute tpus tpu-vm ssh $VM_NAME --zone=$ZONE --project=$PROJECT --worker=0 \\
  --command='curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH=\$HOME/.local/bin:\$PATH && uv venv ~/nopenv --python 3.12 && uv pip install --python ~/nopenv --upgrade \"jax[tpu]\" numpy && ~/nopenv/bin/python -c \"import jax; print(jax.__version__, jax.devices())\"'" ;;
    copyback)  echo "gcloud compute tpus tpu-vm ssh $VM_NAME --zone=$ZONE --project=$PROJECT --worker=0 \\
  --command='cd /tmp && tar czf nop_prof.tar.gz nop_prof' && \\
gcloud compute tpus tpu-vm scp $VM_NAME:/tmp/nop_prof.tar.gz $LOCAL_DIR/traces/ \\
  --zone=$ZONE --project=$PROJECT --worker=0" ;;
    describe)  echo "gcloud compute tpus tpu-vm describe $VM_NAME --zone=$ZONE --project=$PROJECT" ;;
    list)      echo "gcloud compute tpus tpu-vm list --zone=$ZONE --project=$PROJECT" ;;
    accel)     echo "gcloud compute tpus accelerator-types list --zone=$ZONE --project=$PROJECT" ;;
    versions)  echo "gcloud compute tpus tpu-vm versions list --zone=$ZONE --project=$PROJECT" ;;
    project)   echo "gcloud config set project <ID>
gcloud auth login" ;;
    quit)      echo "exit" ;;
  esac
}

# Extra context shown under the command in the preview pane.
note_for() {
  case "$1" in
    create)    echo "Billing starts when the node reaches READY. Refuses to run if a TPU with this name already exists." ;;
    hunt)      echo "Walks the zone list until one has free capacity. The winning zone is saved to .tpu_zone so every later command follows it." ;;
    queue)     echo "Queues the request and waits for capacity instead of failing. Poll with qstatus. Delete it with qdelete, NOT delete." ;;
    start)     echo "Resumes a STOPPED VM. Not supported on spot VMs -- with --spot you delete and recreate instead." ;;
    stop)      echo "Halts chip billing, keeps the boot disk. Not supported on spot VMs." ;;
    delete)    echo "Destroys the VM and its disk, and stops all billing. Copy results back (copyback) first." ;;
    ssh)       echo "Generates and pushes SSH keys on first use." ;;
    run)       echo "Runs the bench on device. Default = bj.compare(case='rfft_1D', ...). Edit at the prompt to run profile()/other cases. Uses the ~/nopenv venv via PYTHONPATH=~/nop." ;;
    upload)    echo "Sends THIS repo's tpu/ tree to ~/nop on the VM (captures/zips/venv excluded). Re-run after editing local code -- nothing here is pulled from git." ;;
    bootstrap) echo "One-time: installs uv + a fresh ~/nopenv venv with the LATEST jax[tpu] + numpy (NOT the old libtpu_releases pin -- that ships an ancient jax where Pallas fails to compile). Does NOT install TensorFlow (parse traces off-box)." ;;
    copyback)  echo "Tars /tmp/nop_prof (the profile() xplane traces) on the VM and pulls it into tpu/traces/ (gitignored). Parse with bj.report() on a box that HAS TensorFlow." ;;
    accel)     echo "If $ACCEL is missing here, this zone has no v5e capacity. Try another zone from the pool." ;;
    project)   echo "Sets the active project and authenticates. Do this once before anything else." ;;
    *)         echo "" ;;
  esac
}

# Preview pane entry point: ./tpu.sh --show <id>
if [ "${1:-}" = "--show" ]; then
  printf '%s%s%s\n' "$grn" "$(cmd_for "${2:-}")" "$rst"
  n="$(note_for "${2:-}")"
  [ -n "$n" ] && printf '\n%s%s%s\n' "$dim" "$n" "$rst"
  exit 0
fi

# ------------------------------------------------------------------- state --
tpu_state() {
  gcloud compute tpus tpu-vm describe "$VM_NAME" \
    --zone="$ZONE" --project="$PROJECT" --format='value(state)' 2>/dev/null </dev/null
}

run() {
  local cmd="$1"
  printf '\n%s%s%s\n' "$dim" "$(printf '%.0s-' {1..72})" "$rst"
  printf '%s%s%s\n' "$grn" "$cmd" "$rst"
  printf '%s%s%s\n' "$dim" "$(printf '%.0s-' {1..72})" "$rst"
  read -r -p "Run this? [y/N] " ok
  case "$ok" in
    y|Y) eval "$cmd"; local rc=$?
         [ $rc -ne 0 ] && printf '%sexit %d%s\n' "$red" "$rc" "$rst"
         return $rc ;;
    *)   echo "skipped." ;;
  esac
}

# ----------------------------------------------------------------- actions --
action_create() {
  local state; state="$(tpu_state)"
  if [ -n "$state" ]; then
    printf '%s%s already exists (state: %s).%s Delete it, or start it.\n' \
      "$ylw" "$VM_NAME" "$state" "$rst"
    return
  fi
  run "$(cmd_for create)"
}

action_hunt() {
  local state; state="$(tpu_state)"
  if [ -n "$state" ]; then
    printf '%s%s already exists (state: %s).%s\n' "$ylw" "$VM_NAME" "$state" "$rst"
    return
  fi
  run "$(cmd_for hunt)" && ZONE="$(cat "$STATE_ZONE" 2>/dev/null || echo "$ZONE")"
}

action_start() {
  [ "$SPOT" = 1 ] && printf '%sSpot VMs cannot be stopped/started -- delete and recreate instead.%s\n' "$ylw" "$rst"
  run "$(cmd_for start)"
}

action_stop() {
  [ "$SPOT" = 1 ] && printf '%sSpot VMs cannot be stopped/started -- delete and recreate instead.%s\n' "$ylw" "$rst"
  run "$(cmd_for stop)"
}

action_delete() {
  printf '%sThis destroys the VM and its disk. Copy results back (copyback) first.%s\n' "$ylw" "$rst"
  run "$(cmd_for delete)"
}

action_run() {
  printf 'Command to run on the TPU %s[default = bench compare]%s: ' "$dim" "$rst"
  read -r remote; [ -z "$remote" ] && remote="$BENCH_DEFAULT"
  run "gcloud compute tpus tpu-vm ssh $VM_NAME --zone=$ZONE --project=$PROJECT --worker=0 --command='$remote'"
}

action_bootstrap() { run "$(cmd_for bootstrap)"; }

action_project() {
  printf 'Current project: %s\n' "$PROJECT"
  printf 'New project id %s[%s]%s: ' "$dim" "$PROJECT" "$rst"
  read -r p
  if [ -n "$p" ] && [ "$p" != "$PROJECT" ]; then
    run "gcloud config set project $p" && PROJECT="$p"
  fi
  printf 'Run auth login? [y/N] '
  read -r a
  case "$a" in y|Y) gcloud auth login ;; esac
}

dispatch() {
  case "$1" in
    create)    action_create ;;
    start)     action_start ;;
    stop)      action_stop ;;
    delete)    action_delete ;;
    run)       action_run ;;
    upload)    run "$(cmd_for upload)" ;;
    bootstrap) action_bootstrap ;;
    copyback)  run "$(cmd_for copyback)" ;;
    project)   action_project ;;
    hunt)      action_hunt ;;
    ssh|queue|qstatus|qdelete|describe|list|accel|versions) run "$(cmd_for "$1")" ;;
    quit)      echo "bye."; exit 0 ;;
  esac
}

# -------------------------------------------------------------------- menu --
MENU=(
  $'create\tcreate     spin up the TPU VM'
  $'hunt\thunt       try every zone until one has capacity'
  $'queue\tqueue      queue the request, wait for capacity'
  $'ssh\tssh        open a shell on the TPU'
  $'run\trun        run the bench on the TPU'
  $'upload\tupload     send this repo tpu/ tree to ~/nop'
  $'bootstrap\tbootstrap  install uv + clone jax[tpu] env'
  $'copyback\tcopyback   pull profile traces back to tpu/traces/'
  $'stop\tstop       halt chip billing, keep the disk'
  $'start\tstart      resume a stopped VM'
  $'delete\tdelete     destroy the VM, stop all billing'
  $'qstatus\tqstatus    state of queued requests'
  $'qdelete\tqdelete    delete a queued resource + its TPU'
  $'describe\tdescribe   full details of this TPU'
  $'list\tlist       all TPUs in this zone'
  $'accel\taccel      accelerator types available in this zone'
  $'versions\tversions   TPU runtime versions in this zone'
  $'project\tproject    set project / gcloud auth login'
  $'quit\tquit       exit'
)

header_text() {
  local state; state="$(tpu_state)"
  [ -z "$state" ] && state="(does not exist)"
  local color="$ylw"
  case "$state" in
    READY) color="$grn" ;;
    "(does not exist)"|STOPPED) color="$dim" ;;
    PREEMPTED|TERMINATED) color="$red" ;;
  esac
  printf '%sproject%s %s   %szone%s %s   %sname%s %s\n%stype%s %s (%s)   %sspot%s %s   %sstate%s %s%s%s' \
    "$cyn" "$rst" "$PROJECT" "$cyn" "$rst" "$ZONE" "$cyn" "$rst" "$VM_NAME" \
    "$cyn" "$rst" "$ACCEL" "$VERSION" "$cyn" "$rst" "$([ "$SPOT" = 1 ] && echo yes || echo no)" \
    "$cyn" "$rst" "$color" "$state" "$rst"
}

pick_fzf() {
  printf '%s\n' "${MENU[@]}" | fzf \
    --ansi --delimiter=$'\t' --with-nth=2.. \
    --height=100% --reverse --cycle --no-sort \
    --prompt='tpu > ' --pointer='>' \
    --header="$(header_text)" --header-first \
    --preview="TPU_PROJECT='$PROJECT' TPU_ZONE='$ZONE' TPU_NAME='$VM_NAME' TPU_ACCEL='$ACCEL' TPU_VERSION='$VERSION' TPU_SPOT='$SPOT' '$SELF' --show {1}" \
    --preview-window='down,8,wrap,border-top' \
  | cut -f1
}

pick_plain() {
  local i=1
  { header_text; echo; } >&2
  for m in "${MENU[@]}"; do printf '%2d  %s\n' "$i" "${m#*$'\t'}" >&2; ((i++)); done
  printf '\nchoice> ' >&2
  read -r n
  [[ "$n" =~ ^[0-9]+$ ]] && [ "$n" -ge 1 ] && [ "$n" -le "${#MENU[@]}" ] \
    && printf '%s' "${MENU[$((n-1))]%%$'\t'*}"
}

while true; do
  clear
  if command -v fzf >/dev/null 2>&1; then id="$(pick_fzf)"; else id="$(pick_plain)"; fi
  [ -z "$id" ] && { echo "bye."; exit 0; }
  dispatch "$id"
  printf '\n%spress enter to return to the menu%s' "$dim" "$rst"
  read -r _
done
