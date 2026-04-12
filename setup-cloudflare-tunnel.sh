#!/bin/bash

# Cloudflare Tunnel Setup Script
# voice-api (port 8000) → voice.<YOUR_DOMAIN>
#
# 사전 준비:
#   1. cloudflared 설치 확인: brew install cloudflared
#   2. DOMAIN 변수에 실제 도메인 입력 후 실행

TUNNEL_NAME="uncounted-voice-api"
PORT=8000
DOMAIN="${DOMAIN:-<YOUR_DOMAIN>}"
SUBDOMAIN="voice"
HOSTNAME="${SUBDOMAIN}.${DOMAIN}"
CREDENTIALS_DIR="$HOME/.cloudflared"

# 0. 도메인 입력 확인
if [ "$DOMAIN" = "<YOUR_DOMAIN>" ]; then
  echo "❌ DOMAIN 변수를 설정해주세요."
  echo "   예: DOMAIN=\"example.com\" bash $0"
  exit 1
fi

echo "=============================="
echo " Cloudflare Tunnel Setup"
echo " Tunnel : $TUNNEL_NAME"
echo " Route  : $HOSTNAME → localhost:$PORT"
echo "=============================="
echo ""

# 1. cloudflared 설치 확인
if ! command -v cloudflared &>/dev/null; then
  echo "❌ cloudflared 가 설치되어 있지 않습니다."
  exit 1
fi
echo "✅ cloudflared $(cloudflared --version 2>&1 | head -1)"

# 2. Cloudflare 로그인
echo ""
echo "▶ Step 1: Cloudflare 로그인"
cloudflared tunnel login

# 3. Tunnel 생성 (이미 있으면 스킵)
echo ""
echo "▶ Step 2: Tunnel 생성 — $TUNNEL_NAME"
if cloudflared tunnel list 2>/dev/null | grep -q "$TUNNEL_NAME"; then
  echo "   ℹ️  이미 존재합니다. 건너뜁니다."
else
  cloudflared tunnel create "$TUNNEL_NAME"
fi

# UUID 추출
TUNNEL_UUID=$(cloudflared tunnel list 2>/dev/null | grep "$TUNNEL_NAME" | awk '{print $1}')
if [ -z "$TUNNEL_UUID" ]; then
  echo "❌ Tunnel UUID를 가져오지 못했습니다."
  exit 1
fi
echo "   UUID: $TUNNEL_UUID"

# 4. config.yml 작성
echo ""
echo "▶ Step 3: config.yml 작성"
mkdir -p "$CREDENTIALS_DIR"
cat > "$CREDENTIALS_DIR/config.yml" << EOF
tunnel: $TUNNEL_UUID
credentials-file: $CREDENTIALS_DIR/$TUNNEL_UUID.json

ingress:
  - hostname: $HOSTNAME
    service: http://localhost:$PORT
  - service: http_status:404
EOF
echo "   완료"
cat "$CREDENTIALS_DIR/config.yml"

# 5. DNS 라우팅
echo ""
echo "▶ Step 4: DNS 라우팅 — $HOSTNAME"
cloudflared tunnel route dns "$TUNNEL_NAME" "$HOSTNAME"

# 6. Tunnel 실행
echo ""
echo "▶ Step 5: Tunnel 실행"
echo "   URL: https://$HOSTNAME → localhost:$PORT"
echo "   종료: Ctrl+C"
echo ""
cloudflared tunnel run "$TUNNEL_NAME"
