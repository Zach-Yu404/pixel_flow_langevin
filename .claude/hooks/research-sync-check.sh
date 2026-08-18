#!/usr/bin/env bash
# Stop hook（Research OS 纪律硬闸）：会话结束前检查
#   1) 是否存在晚于最近一次 .research 同步的代码 commit
#   2) 此时是否有 open 的 状态:review issue（Codex 回路是否被触发）
# 命中则 block 一次（reason 会喂回给 Claude 去补同步/开 issue）；同一会话第二次
# Stop 只提醒不再拦，避免死循环。
set -uo pipefail
input=$(cat 2>/dev/null || echo '{}')
sid=$(echo "$input" | jq -r '.session_id // "nosession"' 2>/dev/null || echo nosession)
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root" || exit 0

last_sync=${RESEARCH_SYNC_FAKE_LAST:-$(git log -1 --format=%ct -- .research 2>/dev/null || echo 0)}
unsynced=$(git log --format='%ct %h %s' -20 -- . ':(exclude).research' 2>/dev/null \
  | awk -v t="$last_sync" '$1 > t {$1=""; print}' | head -5)
open_review=$(timeout 8 ~/.local/bin/gh issue list --label "状态:review" --state open \
  --json number -q 'length' 2>/dev/null || echo "?")

msg=""
[ -n "$unsynced" ] && msg="有代码 commit 晚于最近一次 .research 同步：${unsynced}"
if [ -n "$unsynced" ] && [ "$open_review" = "0" ]; then
  msg="${msg}；且没有 open 的 状态:review issue（Codex 回路未触发）"
fi
[ -z "$msg" ] && exit 0

sent="/tmp/research-sync-block-${sid}"
if [ ! -f "$sent" ]; then
  touch "$sent"
  jq -n --arg r "[research-sync] ${msg} —— 请先补 .research 同步（task/CURRENT/STATE）并按需开 review issue，再结束。（本会话仅拦截一次）" \
    '{decision:"block", reason:$r}'
else
  jq -n --arg m "[research-sync] 提醒：${msg}" '{systemMessage:$m}'
fi
