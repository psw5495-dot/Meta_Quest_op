#!/usr/bin/env python3
"""
tesollo_quest3.py - Meta Quest 3 Hand Tracking → Tesollo DG-5F-M Teleoperation
MediaPipe 완전 제거, Quest 3 OpenXR 데이터 직접 사용
"""

import os, time, math, socket, struct, threading
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════
GRIPPER_IP   = "169.254.186.72"
GRIPPER_PORT = 502
QUEST_TCP_PORT = 7000

CONTROL_HZ = 50
DT = 1.0 / CONTROL_HZ

MOTOR_CONFIG = {
    1:  (1.3, 300, "thumb opposition"),
    2:  (1.2, 320, "thumb CMC flex"),
    3:  (1.0, 280, "thumb MCP flex"),
    4:  (0.8, 200, "thumb IP flex"),
    5:  (1.2, 250, "index spread"),
    6:  (0.8, 200, "index flex 1"),
    7:  (0.8, 200, "index flex 2"),
    8:  (0.8, 200, "index flex 3"),
    9:  (1.2, 250, "middle spread"),
    10: (0.8, 200, "middle flex 1"),
    11: (0.8, 200, "middle flex 2"),
    12: (0.8, 200, "middle flex 3"),
    13: (1.2, 250, "ring spread"),
    14: (0.8, 200, "ring flex 1"),
    15: (0.8, 200, "ring flex 2"),
    16: (0.8, 200, "ring flex 3"),
    17: (1.5, 320, "pinky spread"),
    18: (1.1, 260, "pinky flex 1"),
    19: (0.8, 200, "pinky flex 2"),
    20: (0.8, 200, "pinky flex 3"),
}

FLEX_DEG_DEFAULT    = 90.0
FLEX_DEG_THUMB_CMC  = 85.0
FLEX_DEG_THUMB_MCP  = 100.0
FLEX_DEG_THUMB_IP   = 90.0

MOTOR2_DISTANCE_MIN    = 0.08
MOTOR2_DISTANCE_MAX    = 0.28
MOTOR2_WEIGHT_DISTANCE = 0.6
MOTOR2_WEIGHT_ANGLE    = 0.4

SPLAY_GAIN_DEFAULT    = 1.0
SPLAY_LIMIT_DEFAULT_DEG = 25.0
SPLAY_GAIN_THUMB      = 2.0
SPLAY_LIMIT_THUMB_DEG = 90.0
SMOOTH_ALPHA          = 0.35

DEADBAND_0P1DEG  = 8
MAX_DUTY_STEP    = 40
TOTAL_DUTY_BUDGET = 1500
MAX_ACTIVE_JOINTS = 12
MIN_DUTY_TO_MOVE  = 18
PROTECTED_JOINTS  = {17, 18}

FINGER_ORDER = ["finger1", "finger2", "finger3", "finger4", "finger5"]
JOINT_MAP = {
    "finger1": [1,  2,  3,  4],
    "finger2": [5,  6,  7,  8],
    "finger3": [9,  10, 11, 12],
    "finger4": [13, 14, 15, 16],
    "finger5": [17, 18, 19, 20],
}
TARGET_SIGN = {i: 1 for i in range(1, 21)}
for m in [3, 4, 17, 18]:
    TARGET_SIGN[m] = -1

DUTY_SIGN    = {i: 1 for i in range(1, 21)}
MOTOR_ENABLED = {m: True for m in range(1, 21)}


# ═══════════════════════════════════════════════════════════════════════════════
# 모터 제어 유틸리티 함수들 (기존 tesollo_dev.py 완전 이식)
# ═══════════════════════════════════════════════════════════════════════════════

def make_zero_duty():
    return {m: 0 for m in range(1, 21)}


def disabled_motor_text():
    disabled = [m for m in range(1, 21) if not MOTOR_ENABLED[m]]
    return ",".join(str(m) for m in disabled) if disabled else "None"


def toggle_motor_enable(motor_id, cur_pos, prev_target, prev_duty):
    if not (1 <= motor_id <= 20):
        return f"Invalid motor id: {motor_id}"
    MOTOR_ENABLED[motor_id] = not MOTOR_ENABLED[motor_id]
    hold = int(cur_pos.get(motor_id, prev_target.get(motor_id, 0)))
    prev_target[motor_id] = hold
    prev_duty[motor_id] = 0
    status = "OFF" if not MOTOR_ENABLED[motor_id] else "ON"
    role = MOTOR_CONFIG[motor_id][2]
    return f"Motor {motor_id:02d} ({role}) -> {status}"


def toggle_all_motors(cur_pos, prev_target, prev_duty):
    any_enabled = any(MOTOR_ENABLED[m] for m in range(1, 21))
    new_state = not any_enabled
    for m in range(1, 21):
        MOTOR_ENABLED[m] = new_state
        if not new_state:
            hold = int(cur_pos.get(m, 0))
            prev_target[m] = hold
            prev_duty[m] = 0
    return f"All motors -> {'ON' if new_state else 'OFF'}"


def toggle_finger_motors(finger_num, cur_pos, prev_target, prev_duty):
    if not (1 <= finger_num <= 5):
        return f"Invalid finger number: {finger_num}"
    finger_key = f"finger{finger_num}"
    motor_ids = JOINT_MAP[finger_key]
    finger_names = {1: "Thumb", 2: "Index", 3: "Middle", 4: "Ring", 5: "Pinky"}
    all_enabled = all(MOTOR_ENABLED[m] for m in motor_ids)
    new_state = not all_enabled
    for m in motor_ids:
        MOTOR_ENABLED[m] = new_state
        if not new_state:
            hold = int(cur_pos.get(m, 0))
            prev_target[m] = hold
            prev_duty[m] = 0
    finger_name = finger_names[finger_num]
    motor_list = ",".join(str(m) for m in motor_ids)
    return f"{finger_name} (M{motor_list}) -> {'ON' if new_state else 'OFF'}"


def enforce_motor_enable_mask(cur_pos, **kwargs):
    for m in range(1, 21):
        if MOTOR_ENABLED[m]:
            continue
        hold = int(cur_pos.get(m, 0))
        for key, data_dict in kwargs.items():
            if data_dict is not None:
                if key in ['desired', 'target']:
                    data_dict[m] = hold
                else:
                    data_dict[m] = 0


def clamp_target_0p1deg(motor_id, target_0p1deg):
    limits = {
        1:  (-150, 290),
        2:  (-850, 900),
        3:  (-1500, 290),
        4:  (-900, 900),
        5:  (-200, 310),
        6:  (0, 1150),
        17: (-300, 0),
        18: (-900, 150),
    }
    if motor_id in limits:
        lo, hi = limits[motor_id]
        return int(np.clip(int(target_0p1deg), lo, hi))
    return int(np.clip(int(target_0p1deg), -900, 1150))


def clamp_step_to_current(motor_id, desired, current):
    """목표값을 현재값 기준 최대 변화량으로 제한 (급격한 움직임 방지)"""
    max_step = 100 if motor_id in [1, 5, 9, 13, 17] else 250
    if motor_id == 2:
        max_step = 130
    elif motor_id == 3:
        max_step = 150
    elif motor_id == 4:
        max_step = 200
    return int(np.clip(int(desired), int(current) - max_step, int(current) + max_step))


def rate_limit_target(motor_id, desired, prev):
    max_speed = 80.0 if motor_id in [1, 5, 9, 13, 17] else 100.0
    if motor_id == 2:
        max_speed = 65.0
    elif motor_id == 3:
        max_speed = 70.0
    elif motor_id == 4:
        max_speed = 90.0
    max_delta = max(1, int(max_speed * 10.0 * DT))
    delta = int(desired) - int(prev)
    return int(prev) + int(np.clip(delta, -max_delta, max_delta))


def slew_limit_duty(motor_id, new_duty, prev_duty):
    prev    = int(prev_duty.get(motor_id, 0))
    limited = int(np.clip(int(new_duty), prev - MAX_DUTY_STEP, prev + MAX_DUTY_STEP))
    prev_duty[motor_id] = limited
    return limited


def apply_global_limits(raw_duty_dict):
    duty = dict(raw_duty_dict)

    # 최소 duty 미만은 0으로
    for m in list(duty.keys()):
        if abs(duty[m]) < MIN_DUTY_TO_MOVE:
            duty[m] = 0

    # 최대 활성 관절 수 제한
    active = [(m, abs(v)) for m, v in duty.items() if v != 0]
    if len(active) > MAX_ACTIVE_JOINTS:
        active.sort(key=lambda x: x[1], reverse=True)
        keep = set(m for m in PROTECTED_JOINTS if duty.get(m, 0) != 0)
        for m, _ in active:
            if len(keep) >= MAX_ACTIVE_JOINTS:
                break
            keep.add(m)
        for m in list(duty.keys()):
            if m not in keep:
                duty[m] = 0

    # 총 duty 예산 제한
    total = sum(abs(v) for v in duty.values())
    if total > TOTAL_DUTY_BUDGET and total > 0:
        scale = TOTAL_DUTY_BUDGET / total
        duty  = {m: int(v * scale) for m, v in duty.items()}

    return duty


def to_duty(err_0p1deg, motor_id):
    if abs(err_0p1deg) < DEADBAND_0P1DEG:
        return 0
    kp, lim, _ = MOTOR_CONFIG[motor_id]
    d = int(kp * err_0p1deg)
    d = int(np.clip(d, -lim, lim))
    return DUTY_SIGN[motor_id] * d


def curl_to_flex_deg(curl_now, flex_deg):
    return float(np.clip(curl_now, 0.0, 1.0)) * flex_deg


# ═══════════════════════════════════════════════════════════════════════════════
# 손 추적 계산 함수들
# ═══════════════════════════════════════════════════════════════════════════════

def _angle_deg(v1, v2):
    dot = float(np.dot(v1, v2))
    n1  = float(np.linalg.norm(v1))
    n2  = float(np.linalg.norm(v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return 180.0
    c = np.clip(dot / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(c))


def _curl_from_joint_angle(angle_deg, open_deg=170.0, closed_deg=70.0):
    curl = (open_deg - angle_deg) / (open_deg - closed_deg)
    return float(np.clip(curl, 0.0, 1.0))


def _finger_curl(lms, finger_idx):
    finger_landmarks = {
        "finger1": [1,  2,  3,  4],
        "finger2": [5,  6,  7,  8],
        "finger3": [9,  10, 11, 12],
        "finger4": [13, 14, 15, 16],
        "finger5": [17, 18, 19, 20],
    }
    idx = finger_landmarks[finger_idx]
    pts = [lms[i] for i in idx]
    if finger_idx == "finger1":
        return 0.0  # 엄지는 별도 처리

    mcp, pip, dip, tip = pts
    ang_pip = _angle_deg(mcp - pip, dip - pip)
    ang_dip = _angle_deg(pip - dip, tip - dip)
    avg     = 0.6 * ang_pip + 0.4 * ang_dip
    return _curl_from_joint_angle(avg)


def _thumb_joint_curls(lms):
    """엄지 MCP(Motor3), IP(Motor4) curl 계산"""
    p1, p2, p3, p4 = lms[1], lms[2], lms[3], lms[4]

    ang_mcp  = _angle_deg(p1 - p2, p3 - p2)
    curl_mcp = _curl_from_joint_angle(ang_mcp)

    ang_ip   = _angle_deg(p2 - p3, p4 - p3)
    curl_ip  = _curl_from_joint_angle(ang_ip)

    return curl_mcp, curl_ip


def compute_thumb_cmc_position(lms_np):
    """Motor2 전용 하이브리드 포지션 제어"""
    thumb_tip  = lms_np[4]
    wrist      = lms_np[0]
    index_mcp  = lms_np[5]

    thumb_distance     = np.linalg.norm(thumb_tip - wrist)
    hand_size          = np.linalg.norm(index_mcp - wrist)
    if hand_size < 1e-6:
        hand_size = 1.0

    normalized_distance = thumb_distance / hand_size
    normalized_distance = np.clip(normalized_distance, MOTOR2_DISTANCE_MIN, MOTOR2_DISTANCE_MAX)
    distance_ratio      = (normalized_distance - MOTOR2_DISTANCE_MIN) / (MOTOR2_DISTANCE_MAX - MOTOR2_DISTANCE_MIN)
    distance_ratio      = float(np.clip(distance_ratio, 0.0, 1.0))

    p0, p1, p2 = lms_np[0], lms_np[1], lms_np[2]
    ang_cmc    = _angle_deg(p0 - p1, p2 - p1)
    angle_ratio = _curl_from_joint_angle(ang_cmc, open_deg=155.0, closed_deg=95.0)

    hybrid_ratio = (MOTOR2_WEIGHT_DISTANCE * distance_ratio +
                    MOTOR2_WEIGHT_ANGLE    * angle_ratio)
    return float(np.clip(hybrid_ratio, 0.0, 1.0))


def compute_splay_deg(lms_np):
    def finger_dir_2d(mcp_idx, tip_idx):
        v = lms_np[tip_idx] - lms_np[mcp_idx]
        return np.array([v[0], v[1]], dtype=np.float32)

    def signed_angle_2d(a, b):
        def unit2(v):
            n = float(np.linalg.norm(v))
            return v / (n + 1e-9)
        a = unit2(a)
        b = unit2(b)
        return math.degrees(math.atan2(
            a[0] * b[1] - a[1] * b[0],
            a[0] * b[0] + a[1] * b[1]))

    dirs = {
        "finger2": finger_dir_2d(5,  8),
        "finger3": finger_dir_2d(9,  12),
        "finger4": finger_dir_2d(13, 16),
        "finger5": finger_dir_2d(17, 20),
        "finger1": finger_dir_2d(2,  4),
    }
    base = dirs["finger3"]
    return {
        "finger3": 0.0,
        "finger2": signed_angle_2d(base, dirs["finger2"]),
        "finger4": signed_angle_2d(base, dirs["finger4"]),
        "finger5": signed_angle_2d(base, dirs["finger5"]),
        "finger1": signed_angle_2d(base, dirs["finger1"]),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Quest 3 TCP 수신 서버
# ═══════════════════════════════════════════════════════════════════════════════

class Quest3HandReceiver:
    def __init__(self, port=QUEST_TCP_PORT):
        self.port              = port
        self.latest_landmarks  = None
        self.latest_pinch      = 0.0
        self.running           = True
        self.lock              = threading.Lock()
        self.connected         = False

        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind(("0.0.0.0", port))
        self.server_sock.listen(1)
        self.server_sock.settimeout(1.0)

        self.thread = threading.Thread(target=self._server_loop, daemon=True)
        self.thread.start()
        print(f"[TCP] Quest 3 수신 서버 시작: 포트 {port}")

    def _server_loop(self):
        while self.running:
            try:
                print("[TCP] Quest 3 연결 대기 중...")
                conn, addr = self.server_sock.accept()
                self.connected = True
                print(f"[TCP] Quest 3 연결됨: {addr}")
                self._handle_client(conn)
                self.connected = False
                print("[TCP] Quest 3 연결 종료, 재대기")
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[TCP] 서버 오류: {e}")
                time.sleep(1.0)

    def _handle_client(self, conn):
        conn.settimeout(2.0)
        buffer = ""
        try:
            while self.running:
                data = conn.recv(4096).decode('utf-8', errors='ignore')
                if not data:
                    break
                buffer += data
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    self._parse_hand_data(line.strip())
        except Exception as e:
            print(f"[TCP] 수신 오류: {e}")
        finally:
            conn.close()

    def _parse_hand_data(self, line):
        try:
            if not line.startswith("HAND_DATA,"):
                return
            data_part = line[10:]
            values    = list(map(float, data_part.split(',')))

            # 21개 관절 × 3좌표(63) + pinch(1) = 64개
            if len(values) >= 64:
                landmarks = []
                for i in range(21):
                    x = values[i * 3]
                    y = values[i * 3 + 1]
                    z = values[i * 3 + 2]
                    landmarks.append([x, y, z])
                pinch = values[63]

                with self.lock:
                    self.latest_landmarks = np.array(landmarks, dtype=np.float32)
                    self.latest_pinch     = pinch
        except Exception as e:
            print(f"[TCP] 파싱 오류: {e}")

    def get_latest_data(self):
        with self.lock:
            return self.latest_landmarks, self.latest_pinch

    def stop(self):
        self.running = False
        try:
            self.server_sock.close()
        except:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# DG5FDevClient (기존 tesollo_dev.py 완전 이식)
# ═══════════════════════════════════════════════════════════════════════════════

class DG5FDevClient:
    def __init__(self, ip, port, timeout=0.5):
        self.ip      = ip
        self.port    = port
        self.timeout = timeout
        self.sock    = None

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect((self.ip, self.port))
        print(f"[OK] Tesollo 연결: {self.ip}:{self.port}")

    def close(self):
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
        self.sock = None

    def _recv_exact(self, n):
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Socket closed")
            buf += chunk
        return buf

    def transact(self, cmd, data=b""):
        length = 2 + 1 + len(data)
        pkt    = struct.pack(">H", length) + struct.pack("B", cmd) + data
        self.sock.sendall(pkt)
        resp_len  = struct.unpack(">H", self._recv_exact(2))[0]
        resp_rest = self._recv_exact(resp_len - 2)
        return resp_rest

    def get_positions(self):
        resp = self.transact(0x01, data=bytes([0x01]))
        if not resp or resp[0] != 0x01:
            raise RuntimeError(f"Unexpected response CMD: {resp[0] if resp else None}")
        payload = resp[1:]
        pos = {}
        i   = 0
        while i + 3 <= len(payload):
            jid = payload[i]
            val = struct.unpack(">h", payload[i + 1:i + 3])[0]
            pos[jid] = val
            i += 3
        return pos

    def set_duty(self, duty_by_id):
        data = b""
        for jid in range(1, 21):
            duty  = int(np.clip(int(duty_by_id.get(jid, 0)), -1000, 1000))
            data += struct.pack("B", jid) + struct.pack(">h", duty)
        length = 2 + 1 + len(data)
        self.sock.sendall(struct.pack(">H", length) + struct.pack("B", 0x05) + data)


# ═══════════════════════════════════════════════════════════════════════════════
# 메인 함수
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("═" * 60)
    print("Quest 3 → Tesollo DG-5F-M Teleoperation")
    print(f"TCP 포트  : {QUEST_TCP_PORT}  (adb reverse tcp:{QUEST_TCP_PORT} tcp:{QUEST_TCP_PORT})")
    print(f"Tesollo   : {GRIPPER_IP}:{GRIPPER_PORT}")
    print("Ctrl+C 로 종료")
    print("═" * 60)

    # Quest 3 수신 서버 시작
    hand_receiver = Quest3HandReceiver()

    # Tesollo Hand 연결
    gr = DG5FDevClient(GRIPPER_IP, GRIPPER_PORT, timeout=0.5)
    gr.connect()

    # 제어 상태 초기화
    smooth_curl      = {f: 0.0 for f in FINGER_ORDER}
    smooth_splay     = {f: 0.0 for f in FINGER_ORDER}
    smooth_thumb_cmc = 0.0
    smooth_thumb_mcp = 0.0
    smooth_thumb_ip  = 0.0

    prev_target       = {m: 0 for m in range(1, 21)}
    prev_target_valid = False
    prev_duty         = {m: 0 for m in range(1, 21)}
    last_target       = {m: 0 for m in range(1, 21)}

    no_hand_frames    = 0
    MAX_NO_HAND_FRAMES = 10

    def reset_targets_to_current(cur_pos):
        nonlocal prev_target, prev_target_valid, last_target
        for m in range(1, 21):
            v = int(cur_pos.get(m, 0))
            prev_target[m] = v
            last_target[m] = v
        prev_target_valid = True

    def reset_duty_state():
        for m in range(1, 21):
            prev_duty[m] = 0

    last_time = time.time()

    try:
        while True:
            # ── 주기 맞추기 ──────────────────────────────────────────────────
            now     = time.time()
            elapsed = now - last_time
            if elapsed < DT:
                time.sleep(DT - elapsed)
            last_time = time.time()

            # ── Tesollo 현재 위치 읽기 ────────────────────────────────────────
            try:
                cur_pos = gr.get_positions()
            except Exception as e:
                print(f"\n[WARN] Tesollo 통신 오류: {e}")
                try:
                    gr.set_duty(make_zero_duty())
                except:
                    pass
                reset_duty_state()
                prev_target_valid = False
                continue

            if not prev_target_valid:
                reset_targets_to_current(cur_pos)
                reset_duty_state()

            # ── Quest 3 손 데이터 받기 ────────────────────────────────────────
            landmarks, pinch = hand_receiver.get_latest_data()

            # ── 손 미감지 처리 ────────────────────────────────────────────────
            if landmarks is None:
                no_hand_frames += 1
                if no_hand_frames >= MAX_NO_HAND_FRAMES:
                    gr.set_duty(make_zero_duty())
                    reset_targets_to_current(cur_pos)
                    reset_duty_state()
                status = "연결대기" if not hand_receiver.connected else "손 미감지"
                print(f"\r[INFO] {status} ({no_hand_frames} frames)        ",
                      end="", flush=True)
                continue

            no_hand_frames = 0

            # ── 손 특성 계산 ──────────────────────────────────────────────────
            curls               = {f: _finger_curl(landmarks, f) for f in FINGER_ORDER}
            splay               = compute_splay_deg(landmarks)
            thumb_cmc_position  = compute_thumb_cmc_position(landmarks)
            thumb_mcp_curl, thumb_ip_curl = _thumb_joint_curls(landmarks)

            # ── 스무딩 ────────────────────────────────────────────────────────
            for f in FINGER_ORDER:
                smooth_curl[f]  = (1.0 - SMOOTH_ALPHA) * smooth_curl[f]  + SMOOTH_ALPHA * curls[f]
                smooth_splay[f] = (1.0 - SMOOTH_ALPHA) * smooth_splay[f] + SMOOTH_ALPHA * splay[f]
                curls[f]  = smooth_curl[f]
                splay[f]  = smooth_splay[f]

            smooth_thumb_cmc = (1.0 - SMOOTH_ALPHA) * smooth_thumb_cmc + SMOOTH_ALPHA * thumb_cmc_position
            smooth_thumb_mcp = (1.0 - SMOOTH_ALPHA) * smooth_thumb_mcp + SMOOTH_ALPHA * thumb_mcp_curl
            smooth_thumb_ip  = (1.0 - SMOOTH_ALPHA) * smooth_thumb_ip  + SMOOTH_ALPHA * thumb_ip_curl
            t_cmc, t_mcp, t_ip = smooth_thumb_cmc, smooth_thumb_mcp, smooth_thumb_ip

            # ── 목표 각도 계산 ────────────────────────────────────────────────
            desired = {m: prev_target[m] for m in range(1, 21)}
            for f in FINGER_ORDER:
                j0, j1, j2, j3 = JOINT_MAP[f]
                if f == "finger1":
                    spread_deg = float(np.clip(
                        SPLAY_GAIN_THUMB * splay[f],
                        -SPLAY_LIMIT_THUMB_DEG, SPLAY_LIMIT_THUMB_DEG))
                    cmc_deg = curl_to_flex_deg(t_cmc, FLEX_DEG_THUMB_CMC)
                    mcp_deg = curl_to_flex_deg(t_mcp, FLEX_DEG_THUMB_MCP)
                    ip_deg  = curl_to_flex_deg(t_ip,  FLEX_DEG_THUMB_IP)

                    desired[j0] = clamp_target_0p1deg(j0, TARGET_SIGN[j0] * int(spread_deg * 10))
                    desired[j1] = clamp_target_0p1deg(j1, TARGET_SIGN[j1] * int(cmc_deg   * 10))
                    desired[j2] = clamp_target_0p1deg(j2, TARGET_SIGN[j2] * int(mcp_deg   * 10))
                    desired[j3] = clamp_target_0p1deg(j3, TARGET_SIGN[j3] * int(ip_deg    * 10))
                else:
                    spread_deg = float(np.clip(
                        SPLAY_GAIN_DEFAULT * splay[f],
                        -SPLAY_LIMIT_DEFAULT_DEG, SPLAY_LIMIT_DEFAULT_DEG))
                    flex_deg = curl_to_flex_deg(curls[f], FLEX_DEG_DEFAULT)
                    desired[j0] = clamp_target_0p1deg(j0, TARGET_SIGN[j0] * int(spread_deg * 10))
                    for jx in [j1, j2, j3]:
                        desired[jx] = clamp_target_0p1deg(jx, TARGET_SIGN[jx] * int(flex_deg * 10))

            # ── 제어 파이프라인 ───────────────────────────────────────────────
            for m in range(1, 21):
                desired[m] = clamp_step_to_current(m, desired[m], cur_pos.get(m, 0))

            enforce_motor_enable_mask(
                cur_pos, desired=desired, prev_target=prev_target, prev_duty=prev_duty)

            target = {}
            for m in range(1, 21):
                target[m]      = rate_limit_target(m, desired[m], prev_target[m])
                prev_target[m] = target[m]
            last_target = dict(target)

            enforce_motor_enable_mask(
                cur_pos, target=target, prev_target=prev_target, prev_duty=prev_duty)

            raw = {m: to_duty(target[m] - cur_pos.get(m, 0), m) for m in range(1, 21)}
            enforce_motor_enable_mask(cur_pos, raw=raw)
            raw = apply_global_limits(raw)
            enforce_motor_enable_mask(cur_pos, raw=raw)

            duty = {m: slew_limit_duty(m, raw.get(m, 0), prev_duty) for m in range(1, 21)}
            enforce_motor_enable_mask(cur_pos, duty=duty, prev_duty=prev_duty)

            # ── Tesollo Hand 제어 명령 전송 ───────────────────────────────────
            try:
                gr.set_duty(duty)
            except Exception as e:
                print(f"\n[WARN] 모터 제어 실패: {e}")
                try:
                    gr.set_duty(make_zero_duty())
                except:
                    pass
                reset_targets_to_current(cur_pos)
                reset_duty_state()
                continue

            # ── 상태 출력 ─────────────────────────────────────────────────────
            active = sum(1 for v in duty.values() if v != 0)
            total  = sum(abs(v) for v in duty.values())
            print(
                f"\r[CTRL] pinch={pinch:.2f} "
                f"cmc={t_cmc:.2f} mcp={t_mcp:.2f} ip={t_ip:.2f} | "
                f"idx={curls['finger2']:.2f} mid={curls['finger3']:.2f} "
                f"rng={curls['finger4']:.2f} pnk={curls['finger5']:.2f} | "
                f"active={active} duty={total}    ",
                end="", flush=True)

    except KeyboardInterrupt:
        print("\n[INFO] 종료 중...")
    finally:
        try:
            gr.set_duty(make_zero_duty())
            print("[OK] 모터 정지 완료")
        except:
            pass
        gr.close()
        hand_receiver.stop()
        print("[OK] 정상 종료")


if __name__ == "__main__":
    main()
