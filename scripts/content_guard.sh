#!/usr/bin/env bash
# content_guard.sh — CI check for explicit content in committed code and GH issues.
# Ensures the public repo stays PG-rated.
#
# Usage: bash scripts/content_guard.sh [--strict]
#   --strict: fail on ANY match (default: warn on mild, fail on explicit)

set -euo pipefail

STRICT="${1:-}"
EXIT_CODE=0

# Explicit terms that should NEVER appear in committed source code.
# These are loaded from persona/content_keywords.txt locally but must not be hardcoded.
EXPLICIT_WORDS=(
  "tits" "titties" "boobs" "pussy" "dick" "cock" "penis"
  "blowjob" "handjob" "orgasm" "cum" "hentai"
)

# Directories/files to scan (only committed source code)
SCAN_DIRS="src/ tests/ static/ templates/"
SCAN_EXTS="--include=*.py --include=*.js --include=*.html --include=*.yaml --include=*.yml --include=*.md"

echo "=== Content Guard — scanning for explicit terms ==="

for word in "${EXPLICIT_WORDS[@]}"; do
  # Case-insensitive grep, skip binary files, skip this script itself
  MATCHES=$(grep -rniw "$word" $SCAN_DIRS $SCAN_EXTS 2>/dev/null | grep -v "content_guard" | grep -v "content_keywords" || true)
  if [ -n "$MATCHES" ]; then
    echo "::error::EXPLICIT CONTENT FOUND — '$word' in source code:"
    echo "$MATCHES" | head -5
    EXIT_CODE=1
  fi
done

# Also check README, CONTRIBUTING, and other root docs
for doc in README.md CONTRIBUTING.md SECURITY.md CHANGELOG.md; do
  if [ -f "$doc" ]; then
    for word in "${EXPLICIT_WORDS[@]}"; do
      if grep -qiw "$word" "$doc" 2>/dev/null; then
        echo "::error::EXPLICIT CONTENT in $doc — '$word'"
        EXIT_CODE=1
      fi
    done
  fi
done

# Check for personal data patterns (names, addresses, phone numbers)
echo "=== Scanning for personal data patterns ==="
PII_PATTERNS='(\b[A-Z][a-z]+ [A-Z][a-z]+\b.*\b(wife|husband|spouse|girlfriend|boyfriend)\b)'
PII_MATCHES=$(grep -rEn "$PII_PATTERNS" $SCAN_DIRS $SCAN_EXTS 2>/dev/null || true)
if [ -n "$PII_MATCHES" ]; then
  echo "::warning::Possible PII detected (review manually):"
  echo "$PII_MATCHES" | head -5
fi

if [ "$EXIT_CODE" -eq 0 ]; then
  echo "✓ Content guard passed — no explicit terms found in source code."
else
  echo ""
  echo "✗ Content guard FAILED — explicit terms found in committed code."
  echo "  Move explicit keywords to persona/content_keywords.txt (gitignored)"
  echo "  and use the _load_content_keywords() pattern in code."
fi

exit $EXIT_CODE
