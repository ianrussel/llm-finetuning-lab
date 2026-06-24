#!/usr/bin/env bash
# Extract ONE Jira repo's issues from ThePublicJiraDataset.zip into data/jira_issues.jsonl,
# without a full ~60 GB restore. It streams the gzipped Mongo archive out of the zip directly into a
# throwaway Docker MongoDB, restores only the chosen repo's collection (--nsInclude), exports it to
# JSONL, and tears everything down. Disk use = just that one collection.
#
# Usage:   ./extract_jira.sh /path/to/ThePublicJiraDataset.zip [REPO] [OUT_JSONL]
# Example: ./extract_jira.sh "../../task-resolution-drafting/data/2025-06-23 ThePublicJiraDataset.zip" IntelDAOS data/jira_issues.jsonl
#
# REPO options (small -> large): Mindville, IntelDAOS, SecondLife, MariaDB, JFrog, Hyperledger,
#   Sakai, Spring, JiraEcosystem, Qt, MongoDB, Sonatype, RedHat, Mojang, Jira, Apache.
# Pick a small one first (disk is tight). IntelDAOS (~10k issues) gives decent issue-type coverage.
set -euo pipefail

ZIP="${1:?usage: extract_jira.sh <zip> [REPO] [OUT]}"
REPO="${2:-IntelDAOS}"
OUT="${3:-data/jira_issues.jsonl}"
ARCHIVE="ThePublicJiraDataset/3. DataDump/mongodump-JiraReposAnon.archive"
VOL="$(pwd)/.jira_mongo_tmp"        # mongod data dir (removed at the end)
IMG="mongo:4.4"                     # 4.4 bundles mongod + mongorestore + mongoexport in one image

cleanup() { docker rm -f jira-mongo >/dev/null 2>&1 || true; rm -rf "$VOL"; }
trap cleanup EXIT
mkdir -p "$VOL" "$(dirname "$OUT")"

echo "[1/5] starting throwaway MongoDB ($IMG)"
docker run -d --name jira-mongo -v "$VOL:/data/db" "$IMG" >/dev/null
until docker exec jira-mongo mongo --quiet --eval 'db.runCommand({ping:1}).ok' 2>/dev/null | grep -q 1; do sleep 2; done

echo "[2/5] streaming '$REPO' from the zip into mongorestore (reads the full 5.8 GB archive, ~15 min)"
unzip -p "$ZIP" "$ARCHIVE" | docker exec -i jira-mongo \
  mongorestore --gzip --archive --nsInclude="JiraReposAnon.$REPO" --drop

echo "[3/5] issue count:"
docker exec jira-mongo mongo JiraReposAnon --quiet --eval "print(db['$REPO'].count())"

echo "[4/5] exporting to JSONL"
docker exec jira-mongo mongoexport --db=JiraReposAnon --collection="$REPO" --out=/tmp/out.jsonl --quiet
docker cp jira-mongo:/tmp/out.jsonl "$OUT"

echo "[5/5] done -> $OUT"
echo "rows: $(wc -l < "$OUT")"
echo "sample top-level keys:"; head -1 "$OUT" | python3 -c "import json,sys; print(sorted(json.loads(sys.stdin.readline()).keys())[:15])" 2>/dev/null || true
echo "Now run prepare_data.py (it reads data/jira_issues.jsonl, handles the nested 'fields' object)."
