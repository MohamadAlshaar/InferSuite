#!/usr/bin/env bash
# Unattended chain: wait for the calendar multi-pass to finish, then prep + run the Creative
# Synthesis video task (task_5 product_launch_video_to_json) with clean per-group passes.
# Video prep (yt-dlp + ffmpeg) is held until calendar passes are done so it doesn't pollute their perf.
set -uo pipefail
cd "$(dirname "$0")"
export PATH="$PWD/tools:$PWD/.venv_litellm/bin:$PATH"   # static ffmpeg + yt-dlp
ROOT="external/WildClawBench"
T5="tasks/05_Creative_Synthesis/05_Creative_Synthesis_task_5_product_launch_video_to_json.md"
T5DIR="$ROOT/workspace/05_Creative_Synthesis/task_5_product_launch_video_to_json/exec"

echo "[chain] waiting for calendar passes to finish..."
for i in $(seq 1 600); do grep -q "ALLDONE" /tmp/oc_passes_status 2>/dev/null && break; sleep 10; done
echo "[chain] calendar passes done -> archiving their results"
cp -r runs/passes runs/passes_calendar 2>/dev/null

echo "[chain] downloading task_5 HF data + Apple-event video ..."
( cd "$ROOT" && source .venv/bin/activate && \
  hf download internlm/WildClawBench --repo-type dataset \
    --include "workspace/05_Creative_Synthesis/task_5_product_launch_video_to_json/*" --local-dir . ) 2>&1 | tail -2
mkdir -p "$T5DIR"
if [ ! -f "$T5DIR/recording.mp4" ]; then
  yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]" --merge-output-format mp4 \
    -o "$T5DIR/product_video.%(ext)s" "https://www.youtube.com/watch?v=H3KnMyojEQU" 2>&1 | tail -3
  [ -f "$T5DIR/product_video.mp4" ] || ffmpeg -y -i "$T5DIR"/product_video.f*.mp4 -i "$T5DIR"/product_video.f*.m4a -c copy "$T5DIR/product_video.mp4" 2>/dev/null || \
    ffmpeg -y -i "$T5DIR"/product_video.* -c copy "$T5DIR/product_video.mp4" 2>/dev/null
  mv -f "$T5DIR/product_video.mp4" "$T5DIR/recording.mp4" 2>/dev/null
  rm -f "$T5DIR"/product_video.f*.* 2>/dev/null
fi
echo "[chain] video ready: $(ls -lh "$T5DIR/recording.mp4" 2>/dev/null | awk '{print $5}')"
[ -f "$T5DIR/recording.mp4" ] || { echo "[chain] FATAL: video prep failed"; exit 1; }

echo "[chain] running creative task_5 clean per-group passes ..."
rm -f runs/passes/group_*.txt runs/passes/freq_*
MODEL=claude-sonnet-4-6 REPEATS=1 bash run_all_passes.sh "$T5"
cp -r runs/passes runs/passes_creative 2>/dev/null
echo "[chain] ALL DONE -> runs/passes_calendar + runs/passes_creative"
