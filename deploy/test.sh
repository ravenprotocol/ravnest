#!/bin/bash
# Smoke test for Ravnest distributed inference demo
set -e

API_URL="${API_URL:-http://localhost:8000}"
PASS=0
FAIL=0

check() {
    local name="$1"
    local expected_code="$2"
    local actual_code="$3"
    if [ "$actual_code" = "$expected_code" ]; then
        echo "PASS: $name (HTTP $actual_code)"
        PASS=$((PASS + 1))
    else
        echo "FAIL: $name (expected HTTP $expected_code, got $actual_code)"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== Ravnest Smoke Tests ==="
echo "API: $API_URL"
echo ""

# 1. Health check
echo "--- Health Check ---"
CODE=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/health")
check "Health endpoint" "200" "$CODE"

# 2. Happy path - send a prompt, get a completion
echo "--- Happy Path ---"
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$API_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "ravnest",
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": 10
    }')
CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | head -n -1)
check "Chat completion" "200" "$CODE"

# Verify response has choices
if echo "$BODY" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['choices'][0]['message']['content']" 2>/dev/null; then
    echo "PASS: Response has content"
    PASS=$((PASS + 1))
else
    echo "FAIL: Response missing content"
    FAIL=$((FAIL + 1))
fi

# 3. Empty messages - should get 400
echo "--- Empty Messages ---"
CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model": "ravnest", "messages": [], "max_tokens": 10}')
check "Empty messages" "400" "$CODE"

# 4. Concurrent request - should get 503
echo "--- Concurrent Request ---"
# Start a long request in background
curl -s -o /dev/null -X POST "$API_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model": "ravnest", "messages": [{"role": "user", "content": "Write a long story about a cat."}], "max_tokens": 50}' &
BG_PID=$!
sleep 1
# Try a second request while first is running
CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model": "ravnest", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 5}')
check "Concurrent request rejected" "503" "$CODE"
wait $BG_PID 2>/dev/null || true

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
