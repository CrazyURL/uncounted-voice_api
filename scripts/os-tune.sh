#!/bin/bash
# ================================================================
# Voice API OS Tuning Script (WSL2 + RTX 4060 Ti)
# ================================================================
# Usage: sudo bash scripts/os-tune.sh
#
# WSL2 제약: systemd는 wsl.conf으로 활성화된 경우만 동작
#            nvidia-smi 클럭 고정은 WSL2에서 제한적
# ================================================================
set -e

echo "========================================"
echo " Voice API OS Tuning (WSL2)"
echo "========================================"

# ── 0. 사전 확인 ──
if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: root 권한 필요 — sudo bash $0"
    exit 1
fi

# ── 1. /dev/shm 임시파일 정리 ──
echo ""
echo "[1/6] /dev/shm 임시파일 정리..."

TEMP_DEV="/dev/shm/stt-temp-dev"
TEMP_LIVE="/dev/shm/stt-temp-live"
RESULTS_DEV="/dev/shm/stt-results-dev"
TEST_FILES=("/dev/shm/test_input.raw" "/dev/shm/test_output.raw")

before=$(du -sh /dev/shm 2>/dev/null | awk '{print $1}')

# 임시 업로드 파일 정리 (처리 완료된 것들)
for dir in "$TEMP_DEV" "$TEMP_LIVE"; do
    if [ -d "$dir" ]; then
        count=$(find "$dir" -type f -mmin +30 | wc -l)
        if [ "$count" -gt 0 ]; then
            find "$dir" -type f -mmin +30 -delete
            echo "  $dir: $count개 파일 삭제 (30분 이상 경과)"
        fi
    fi
done

# 완료된 결과 디렉토리 정리 (1시간 이상)
for dir in "$RESULTS_DEV"; do
    if [ -d "$dir" ]; then
        count=$(find "$dir" -mindepth 1 -maxdepth 1 -type d -mmin +60 | wc -l)
        if [ "$count" -gt 0 ]; then
            find "$dir" -mindepth 1 -maxdepth 1 -type d -mmin +60 -exec rm -rf {} +
            echo "  $dir: $count개 결과 디렉토리 삭제 (1시간 이상 경과)"
        fi
    fi
done

# 테스트 파일 정리
for f in "${TEST_FILES[@]}"; do
    if [ -f "$f" ]; then
        rm -f "$f"
        echo "  테스트 파일 삭제: $f"
    fi
done

# stale loky 세마포어 정리
stale_sem=$(find /dev/shm -name 'sem.loky-*' -mmin +60 2>/dev/null | wc -l)
if [ "$stale_sem" -gt 0 ]; then
    find /dev/shm -name 'sem.loky-*' -mmin +60 -delete 2>/dev/null
    echo "  stale loky 세마포어 $stale_sem개 삭제"
fi

after=$(du -sh /dev/shm 2>/dev/null | awk '{print $1}')
echo "  /dev/shm: $before → $after"

# ── 2. sysctl 네트워크 + 메모리 튜닝 ──
echo ""
echo "[2/6] sysctl 튜닝..."

cat > /etc/sysctl.d/99-voice-api.conf << 'SYSCTL'
# Voice API OS Tuning — Network
net.core.netdev_max_backlog = 5000
net.ipv4.tcp_max_syn_backlog = 4096
net.ipv4.tcp_tw_reuse = 1
net.ipv4.tcp_fin_timeout = 15
net.ipv4.ip_local_port_range = 10240 65535

# TCP Buffers — 대용량 오디오 업로드
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.ipv4.tcp_rmem = 4096 87380 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216

# Memory — OOM 방어
vm.swappiness = 10
vm.overcommit_memory = 0
vm.dirty_ratio = 20
vm.dirty_background_ratio = 5
SYSCTL

sysctl --system > /dev/null 2>&1
echo "  /etc/sysctl.d/99-voice-api.conf 생성 및 적용 완료"

# 적용 확인
echo "  vm.swappiness = $(sysctl -n vm.swappiness)"
echo "  net.ipv4.tcp_fin_timeout = $(sysctl -n net.ipv4.tcp_fin_timeout)"
echo "  net.ipv4.tcp_tw_reuse = $(sysctl -n net.ipv4.tcp_tw_reuse)"

# ── 3. ulimits ──
echo ""
echo "[3/6] ulimits 설정..."

cat > /etc/security/limits.d/voice-api.conf << 'LIMITS'
# Voice API — file descriptors & process limits
gdash  soft  nofile  65536
gdash  hard  nofile  65536
gdash  soft  nproc   4096
gdash  hard  nproc   4096
gdash  soft  memlock unlimited
gdash  hard  memlock unlimited
LIMITS

echo "  /etc/security/limits.d/voice-api.conf 생성 완료"
echo "  (새 세션부터 적용됨)"

# ── 4. nvidia-smi PATH 등록 ──
echo ""
echo "[4/6] nvidia-smi PATH 등록..."

NVIDIA_PROFILE="/etc/profile.d/nvidia-wsl.sh"
if [ ! -f "$NVIDIA_PROFILE" ]; then
    cat > "$NVIDIA_PROFILE" << 'NVPATH'
# WSL2 nvidia-smi PATH
if [ -d /usr/lib/wsl/lib ]; then
    export PATH="/usr/lib/wsl/lib:$PATH"
    export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
fi
NVPATH
    chmod 644 "$NVIDIA_PROFILE"
    echo "  $NVIDIA_PROFILE 생성 완료"
else
    echo "  $NVIDIA_PROFILE 이미 존재"
fi

# 현재 세션에도 즉시 적용
export PATH="/usr/lib/wsl/lib:$PATH"

# ── 5. /dev/shm 크기 확대 ──
echo ""
echo "[5/6] /dev/shm 크기 확인..."

shm_size=$(df -BG /dev/shm | tail -1 | awk '{print $2}')
echo "  현재 /dev/shm 크기: $shm_size"

# WSL2에서는 /dev/shm 기본이 RAM 50%
# 필요 시 remount (WSL2는 재부팅하면 리셋됨)
shm_avail=$(df -BM /dev/shm | tail -1 | awk '{gsub("M","",$4); print $4}')
if [ "$shm_avail" -lt 1024 ]; then
    echo "  WARNING: /dev/shm 가용 공간 ${shm_avail}MB — 임시파일 정리 후 재확인"
fi

# ── 6. 자동 정리 cron ──
echo ""
echo "[6/6] 자동 정리 cron 등록..."

CRON_SCRIPT="/usr/local/bin/voice-api-cleanup.sh"
cat > "$CRON_SCRIPT" << 'CLEANUP'
#!/bin/bash
# Voice API — /dev/shm 자동 정리 (매시간 실행)
for dir in /dev/shm/stt-temp-dev /dev/shm/stt-temp-live; do
    [ -d "$dir" ] && find "$dir" -type f -mmin +60 -delete 2>/dev/null
done
for dir in /dev/shm/stt-results-dev /dev/shm/stt-results-live; do
    [ -d "$dir" ] && find "$dir" -mindepth 1 -maxdepth 1 -type d -mmin +120 -exec rm -rf {} + 2>/dev/null
done
find /dev/shm -name 'sem.loky-*' -mmin +60 -delete 2>/dev/null
CLEANUP
chmod +x "$CRON_SCRIPT"

# crontab에 등록 (중복 방지)
CRON_LINE="0 * * * * $CRON_SCRIPT"
(crontab -l 2>/dev/null | grep -v "voice-api-cleanup" ; echo "$CRON_LINE") | crontab -
echo "  $CRON_SCRIPT 생성 + crontab 매시간 실행 등록 완료"

# ── 최종 요약 ──
echo ""
echo "========================================"
echo " 튜닝 완료 — 최종 상태"
echo "========================================"
echo ""
echo "  /dev/shm:"
df -h /dev/shm | tail -1 | awk '{printf "    크기: %s  사용: %s  가용: %s  (%s)\n", $2, $3, $4, $5}'
echo ""
echo "  sysctl:"
echo "    vm.swappiness = $(sysctl -n vm.swappiness)"
echo "    tcp_fin_timeout = $(sysctl -n net.ipv4.tcp_fin_timeout)"
echo "    tcp_tw_reuse = $(sysctl -n net.ipv4.tcp_tw_reuse)"
echo "    ip_local_port_range = $(sysctl -n net.ipv4.ip_local_port_range)"
echo ""
echo "  GPU:"
if command -v nvidia-smi &>/dev/null || [ -x /usr/lib/wsl/lib/nvidia-smi ]; then
    /usr/lib/wsl/lib/nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader 2>/dev/null || echo "    조회 실패"
fi
echo ""
echo "  다음 단계:"
echo "    1. 새 터미널 열어서 ulimit -n 확인 (65536이어야 함)"
echo "    2. voice-api 서비스 재시작: sudo systemctl restart voice-api@dev"
echo "    3. run.sh 환경변수 보강 (별도 패치)"
echo "========================================"
