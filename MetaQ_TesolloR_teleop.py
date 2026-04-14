#!/usr/bin/env python3
"""
tesollo_quest3.py - Meta Quest 3 Hand Tracking → Tesollo DG-5F-M Teleoperation
MediaPipe 완전 제거, Quest 3 OpenXR 데이터 직접 사용 + 실시간 GUI 시각화
"""

import os, time, math, socket, struct, threading, sys
import numpy as np
import tkinter as tk
from tkinter import ttk

if os.name == "nt":
    import msvcrt
else:
    import select
    import termios
    import tty


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


def _nonthumb_joint_curls(lms, finger_idx):
    finger_landmarks = {
        "finger2": [5,  6,  7,  8],   # index
        "finger3": [9,  10, 11, 12],  # middle
        "finger4": [13, 14, 15, 16],  # ring
        "finger5": [17, 18, 19, 20],  # pinky
    }

    idx = finger_landmarks[finger_idx]
    mcp, pip, dip, tip = [lms[i] for i in idx]
    wrist = lms[0]

    ang_mcp = _angle_deg(wrist - mcp, pip - mcp)
    ang_pip = _angle_deg(mcp - pip, dip - pip)
    ang_dip = _angle_deg(pip - dip, tip - dip)

    curl_mcp = _curl_from_joint_angle(ang_mcp, open_deg=165.0, closed_deg=70.0)
    curl_pip = _curl_from_joint_angle(ang_pip, open_deg=170.0, closed_deg=70.0)
    curl_dip = _curl_from_joint_angle(ang_dip, open_deg=170.0, closed_deg=80.0)

    return curl_mcp, curl_pip, curl_dip


def _thumb_joint_curls(lms):
    p1, p2, p3, p4 = lms[1], lms[2], lms[3], lms[4]

    ang_mcp  = _angle_deg(p1 - p2, p3 - p2)
    curl_mcp = _curl_from_joint_angle(ang_mcp)

    ang_ip   = _angle_deg(p2 - p3, p4 - p3)
    curl_ip  = _curl_from_joint_angle(ang_ip)

    return curl_mcp, curl_ip


def compute_thumb_cmc_position(lms_np):
    thumb_tip  = lms_np[4]
    wrist      = lms_np[0]
    index_mcp  = lms_np[5]

    thumb_distance = np.linalg.norm(thumb_tip - wrist)
    hand_size      = np.linalg.norm(index_mcp - wrist)
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


def compute_pinky_cmc_position(lms_np, splay_deg_dict):
    pinky_splay = splay_deg_dict["finger5"]
    cmc_proxy_deg = np.clip(0.6 * pinky_splay, -20.0, 20.0)
    ratio = (cmc_proxy_deg + 20.0) / 40.0
    return float(np.clip(ratio, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════════════
# 모터 제어 유틸리티 함수들
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
    all_enabled = all(MOTOR_ENABLED[m] for m in range(1, 21))
    new_state = not all_enabled

    for m in range(1, 21):
        MOTOR_ENABLED[m] = new_state
        hold = int(cur_pos.get(m, prev_target.get(m, 0)))
        prev_target[m] = hold
        prev_duty[m] = 0

    return f"All motors -> {'ON' if new_state else 'OFF'}"


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

    for m in list(duty.keys()):
        if abs(duty[m]) < MIN_DUTY_TO_MOVE:
            duty[m] = 0

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
# 실시간 GUI 시각화 클래스
# ═══════════════════════════════════════════════════════════════════════════════

class HandJointVisualizer:
    """별도 스레드에서 실행되는 관절 각도 시각화 GUI"""

    def __init__(self):
        self.lock = threading.Lock()
        self.running = True
        self.data = {
            "thumb": {"cmc": 0.0, "mcp": 0.0, "ip": 0.0},
            "fingers": {
                "finger2": [0.0, 0.0, 0.0],
                "finger3": [0.0, 0.0, 0.0],
                "finger4": [0.0, 0.0, 0.0],
                "finger5": [0.0, 0.0, 0.0],
            },
            "splay": {f: 0.0 for f in FINGER_ORDER},
            "pinch": 0.0,
            "status": "대기중",
            "active": 0,
            "duty": 0,
        }

        self.thread = threading.Thread(target=self._run_gui, daemon=True)
        self.thread.start()

    def update_data(self, **kwargs):
        """메인 루프에서 호출하여 데이터 업데이트"""
        with self.lock:
            self.data.update(kwargs)

    def _run_gui(self):
        """GUI 메인 루프 (별도 스레드)"""
        self.root = tk.Tk()
        self.root.title("Hand Joint Angles - Quest 3 → Tesollo")
        self.root.geometry("620x580")
        self.root.configure(bg="#2b2b2b")

        self._create_widgets()
        self._schedule_update()

        try:
            self.root.mainloop()
        except:
            pass

    def _create_widgets(self):
        """GUI 위젯 생성"""
        # 상단 상태 표시
        status_frame = tk.Frame(self.root, bg="#313244", pady=8)
        status_frame.pack(fill="x", padx=5, pady=5)

        self.status_label = tk.Label(status_frame, text="● 대기중",
                                   fg="#ff6b6b", bg="#313244", 
                                   font=("Consolas", 12, "bold"))
        self.status_label.pack(side="left", padx=15)

        self.pinch_label = tk.Label(status_frame, text="Pinch: 0.00",
                                  fg="#4ecdc4", bg="#313244", 
                                  font=("Consolas", 12))
        self.pinch_label.pack(side="left", padx=15)

        self.active_label = tk.Label(status_frame, text="Active: 0  Duty: 0",
                                   fg="#fab387", bg="#313244", 
                                   font=("Consolas", 12))
        self.active_label.pack(side="right", padx=15)

        # 구분선
        tk.Frame(self.root, bg="#555", height=2).pack(fill="x", pady=5)

        # 스크롤 가능한 메인 영역
        canvas = tk.Canvas(self.root, bg="#2b2b2b", highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#2b2b2b")

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        canvas.pack(side="left", fill="both", expand=True, padx=(10,0))
        scrollbar.pack(side="right", fill="y", padx=(0,10))

        # 손가락별 위젯 생성
        self.progress_bars = {}
        self.value_labels = {}

        finger_names = {"finger1": "Thumb", "finger2": "Index", "finger3": "Middle", 
                       "finger4": "Ring", "finger5": "Pinky"}
        colors = {"finger1": "#ff6b6b", "finger2": "#4ecdc4", "finger3": "#45b7d1",
                 "finger4": "#96ceb4", "finger5": "#ffeaa7"}

        for i, (finger_key, finger_name) in enumerate(finger_names.items()):
            color = colors[finger_key]

            # 손가락 제목
            title_frame = tk.Frame(scrollable_frame, bg="#2b2b2b")
            title_frame.pack(fill="x", pady=(10 if i > 0 else 0, 5))

            tk.Label(title_frame, text=f"▶ {finger_name}", fg=color, bg="#2b2b2b",
                    font=("Consolas", 11, "bold")).pack(side="left")

            # Splay 표시
            splay_frame = tk.Frame(scrollable_frame, bg="#2b2b2b")
            splay_frame.pack(fill="x", padx=20, pady=2)

            tk.Label(splay_frame, text="Splay:", fg="#999", bg="#2b2b2b",
                    font=("Consolas", 9), width=8, anchor="w").pack(side="left")

            splay_bar = ttk.Progressbar(splay_frame, length=220, mode='determinate')
            splay_bar.pack(side="left", padx=5)

            splay_label = tk.Label(splay_frame, text="0.0°", fg="#ccc", bg="#2b2b2b",
                                 font=("Consolas", 9), width=8)
            splay_label.pack(side="left", padx=5)

            self.progress_bars[(finger_key, "splay")] = splay_bar
            self.value_labels[(finger_key, "splay")] = splay_label

            # 관절별 막대
            if finger_key == "finger1":
                joints = [("cmc", "CMC"), ("mcp", "MCP"), ("ip", "IP")]
            else:
                joints = [("mcp", "MCP"), ("pip", "PIP"), ("dip", "DIP")]

            for joint_key, joint_name in joints:
                joint_frame = tk.Frame(scrollable_frame, bg="#2b2b2b")
                joint_frame.pack(fill="x", padx=30, pady=1)

                tk.Label(joint_frame, text=f"{joint_name}:", fg="#999", bg="#2b2b2b",
                        font=("Consolas", 9), width=6, anchor="w").pack(side="left")

                progress_bar = ttk.Progressbar(joint_frame, length=220, mode='determinate')
                progress_bar.pack(side="left", padx=5)

                value_label = tk.Label(joint_frame, text="0.00", fg="#ccc", bg="#2b2b2b",
                                     font=("Consolas", 9), width=6)
                value_label.pack(side="left", padx=5)

                self.progress_bars[(finger_key, joint_key)] = progress_bar
                self.value_labels[(finger_key, joint_key)] = value_label

    def _schedule_update(self):
        """30Hz로 GUI 업데이트"""
        if not self.running:
            return

        try:
            self._update_display()
        except:
            pass

        self.root.after(33, self._schedule_update)  # ~30Hz

    def _update_display(self):
        """화면 업데이트"""
        with self.lock:
            data_copy = dict(self.data)

        # 상태 업데이트
        pinch = data_copy.get("pinch", 0.0)
        status = data_copy.get("status", "대기중")
        active = data_copy.get("active", 0)
        duty = data_copy.get("duty", 0)

        if pinch > 0.5:
            self.status_label.config(text="● PINCH", fg="#a6e3a1")
        elif status == "추적중":
            self.status_label.config(text="● 추적중", fg="#4ecdc4")
        else:
            self.status_label.config(text="● 대기중", fg="#ff6b6b")

        self.pinch_label.config(text=f"Pinch: {pinch:.2f}")
        self.active_label.config(text=f"Active: {active}  Duty: {duty}")

        # 엄지 업데이트
        thumb_data = data_copy.get("thumb", {})
        for joint in ["cmc", "mcp", "ip"]:
            value = thumb_data.get(joint, 0.0)
            key = ("finger1", joint)
            if key in self.progress_bars:
                self.progress_bars[key]['value'] = value * 100
                self.value_labels[key].config(text=f"{value:.2f}")

        # 다른 손가락 업데이트
        fingers_data = data_copy.get("fingers", {})
        for finger_key in ["finger2", "finger3", "finger4", "finger5"]:
            joint_values = fingers_data.get(finger_key, [0.0, 0.0, 0.0])
            joint_names = ["mcp", "pip", "dip"]

            for i, joint_name in enumerate(joint_names):
                if i < len(joint_values):
                    value = joint_values[i]
                    key = (finger_key, joint_name)
                    if key in self.progress_bars:
                        self.progress_bars[key]['value'] = value * 100
                        self.value_labels[key].config(text=f"{value:.2f}")

        # Splay 업데이트
        splay_data = data_copy.get("splay", {})
        for finger_key in FINGER_ORDER:
            splay_value = splay_data.get(finger_key, 0.0)
            key = (finger_key, "splay")
            if key in self.progress_bars:
                # -30~30도를 0~100으로 매핑
                normalized = (splay_value + 30.0) / 60.0 * 100
                self.progress_bars[key]['value'] = max(0, min(100, normalized))
                self.value_labels[key].config(text=f"{splay_value:+.1f}°")

    def stop(self):
        """GUI 종료"""
        self.running = False
        if hasattr(self, 'root'):
            try:
                self.root.quit()
            except:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
# DG5FDevClient (Tesollo 통신)
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
    print("Quest 3 → Tesollo DG-5F-M Teleoperation (실시간 GUI 포함)")
    print(f"TCP 포트  : {QUEST_TCP_PORT}  (adb reverse tcp:{QUEST_TCP_PORT} tcp:{QUEST_TCP_PORT})")
    print(f"Tesollo   : {GRIPPER_IP}:{GRIPPER_PORT}")
    print("Ctrl+C 로 종료")
    print("═" * 60)

    # Quest 3 수신 서버 시작
    hand_receiver = Quest3HandReceiver()

    # 관절 각도 시각화 GUI 시작
    visualizer = HandJointVisualizer()

    # Tesollo Hand 연결
    gr = DG5FDevClient(GRIPPER_IP, GRIPPER_PORT, timeout=0.5)
    gr.connect()

    # 제어 상태 초기화
    smooth_joint_curls = {
        "finger2": [0.0, 0.0, 0.0],
        "finger3": [0.0, 0.0, 0.0],
        "finger4": [0.0, 0.0, 0.0],
        "finger5": [0.0, 0.0, 0.0],
    }
    smooth_splay = {f: 0.0 for f in FINGER_ORDER}

    smooth_thumb_cmc = 0.0
    smooth_thumb_mcp = 0.0
    smooth_thumb_ip  = 0.0

    smooth_pinky_cmc = 0.0

    prev_target       = {m: 0 for m in range(1, 21)}
    prev_target_valid = False
    prev_duty         = {m: 0 for m in range(1, 21)}
    last_target       = {m: 0 for m in range(1, 21)}

    cur_pos = {m: 0 for m in range(1, 21)}

    state_lock = threading.Lock()

    def reset_duty_state():
        for m in range(1, 21):
            prev_duty[m] = 0

    def reset_targets_to_current(cur_pos_local):
        nonlocal prev_target_valid, last_target
        for m in range(1, 21):
            v = int(cur_pos_local.get(m, 0))
            prev_target[m] = v
            last_target[m] = v
        prev_target_valid = True

    # 키보드 입력 스레드
    def keyboard_input_loop():
        nonlocal cur_pos, prev_target, prev_duty

        print("\n[KEY] 실시간 키 입력 모드 시작")
        print("  T  : 모든 모터 ON/OFF 토글")
        print("  Q  : 키 입력 스레드 종료")
        print("  3~9 : 즉시 해당 모터 토글")
        print("  1,2 : 0.55초 내 다음 숫자와 조합해서 10~20 해석 가능 (예: 1+7=>17)\n")

        digit_buffer = ""
        digit_time = 0.0
        digit_timeout = 0.55
        running = True

        def execute_motor_toggle(motor_id):
            with state_lock:
                msg = toggle_motor_enable(motor_id, cur_pos, prev_target, prev_duty)
            print(f"\n[KEY] {msg}")
            print(f"[KEY] Disabled motors: {disabled_motor_text()}")

        def execute_all_toggle():
            with state_lock:
                msg = toggle_all_motors(cur_pos, prev_target, prev_duty)
            print(f"\n[KEY] {msg}")
            print(f"[KEY] Disabled motors: {disabled_motor_text()}")

        def flush_digit_buffer(force=False):
            nonlocal digit_buffer, digit_time
            if digit_buffer == "":
                return
            if force or (time.time() - digit_time >= digit_timeout):
                try:
                    motor_id = int(digit_buffer)
                    if 1 <= motor_id <= 20:
                        execute_motor_toggle(motor_id)
                    else:
                        print(f"\n[KEY] 잘못된 모터 번호: {digit_buffer}")
                finally:
                    digit_buffer = ""
                    digit_time = 0.0

        def process_digit(ch):
            nonlocal digit_buffer, digit_time
            now = time.time()

            if digit_buffer != "":
                candidate = digit_buffer + ch
                if candidate.isdigit() and 1 <= int(candidate) <= 20:
                    motor_id = int(candidate)
                    digit_buffer = ""
                    digit_time = 0.0
                    execute_motor_toggle(motor_id)
                    return

                old = digit_buffer
                digit_buffer = ""
                digit_time = 0.0
                if old.isdigit():
                    old_id = int(old)
                    if 1 <= old_id <= 20:
                        execute_motor_toggle(old_id)

                process_digit(ch)
                return

            if ch in "3456789":
                execute_motor_toggle(int(ch))
                return

            if ch in "012":
                digit_buffer = ch
                digit_time = now
                return

        def handle_key(ch):
            nonlocal running
            if not ch:
                return
            ch = ch.upper()

            if ch == "T":
                flush_digit_buffer(force=True)
                execute_all_toggle()
                return
            if ch == "Q":
                flush_digit_buffer(force=True)
                print("\n[KEY] 실시간 키 입력 스레드 종료")
                running = False
                return
            if ch.isdigit():
                process_digit(ch)
                return

        if os.name == "nt":
            while running:
                flush_digit_buffer(force=False)
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    if ch in ("\x00", "\xe0"):
                        if msvcrt.kbhit():
                            msvcrt.getwch()
                        continue
                    handle_key(ch)
                time.sleep(0.01)
        else:
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                while running:
                    flush_digit_buffer(force=False)
                    rlist, _, _ = select.select([sys.stdin], [], [], 0.01)
                    if rlist:
                        ch = sys.stdin.read(1)
                        handle_key(ch)
                    time.sleep(0.005)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    key_thread = threading.Thread(target=keyboard_input_loop, daemon=True)
    key_thread.start()

    # 메인 루프
    no_hand_frames = 0
    MAX_NO_HAND_FRAMES = 10
    last_time = time.time()

    try:
        while True:
            # 주기 맞추기
            now = time.time()
            elapsed = now - last_time
            if elapsed < DT:
                time.sleep(DT - elapsed)
            last_time = time.time()

            # Tesollo 현재 위치 읽기
            try:
                new_pos = gr.get_positions()
                with state_lock:
                    cur_pos = new_pos
            except Exception as e:
                print(f"\n[WARN] Tesollo 통신 오류: {e}")
                try:
                    gr.set_duty(make_zero_duty())
                except:
                    pass
                with state_lock:
                    reset_duty_state()
                    prev_target_valid = False
                visualizer.update_data(status="대기중")
                continue

            with state_lock:
                if not prev_target_valid:
                    reset_targets_to_current(cur_pos)
                    reset_duty_state()

            # Quest 손 데이터 받기
            landmarks, pinch = hand_receiver.get_latest_data()

            # 손 미감지 처리
            if landmarks is None:
                no_hand_frames += 1
                if no_hand_frames >= MAX_NO_HAND_FRAMES:
                    try:
                        gr.set_duty(make_zero_duty())
                    except:
                        pass
                    with state_lock:
                        reset_targets_to_current(cur_pos)
                        reset_duty_state()

                status = "연결대기" if not hand_receiver.connected else "손 미감지"
                visualizer.update_data(status=status, pinch=0.0)
                print(f"\r[INFO] {status} ({no_hand_frames} frames)        ", end="", flush=True)
                continue

            no_hand_frames = 0

            # 1) 손 특성 계산(원시값)
            splay = compute_splay_deg(landmarks)

            thumb_cmc_position = compute_thumb_cmc_position(landmarks)
            thumb_mcp_curl, thumb_ip_curl = _thumb_joint_curls(landmarks)

            joint_curls = {
                "finger2": _nonthumb_joint_curls(landmarks, "finger2"),
                "finger3": _nonthumb_joint_curls(landmarks, "finger3"),
                "finger4": _nonthumb_joint_curls(landmarks, "finger4"),
                "finger5": _nonthumb_joint_curls(landmarks, "finger5"),
            }

            pinky_cmc_position = compute_pinky_cmc_position(landmarks, splay)

            # 2) 스무딩
            for f in FINGER_ORDER:
                smooth_splay[f] = (1.0 - SMOOTH_ALPHA) * smooth_splay[f] + SMOOTH_ALPHA * splay[f]
                splay[f] = smooth_splay[f]

            for f in ["finger2", "finger3", "finger4", "finger5"]:
                mcp_c, pip_c, dip_c = joint_curls[f]
                prev = smooth_joint_curls[f]
                prev[0] = (1.0 - SMOOTH_ALPHA) * prev[0] + SMOOTH_ALPHA * mcp_c
                prev[1] = (1.0 - SMOOTH_ALPHA) * prev[1] + SMOOTH_ALPHA * pip_c
                prev[2] = (1.0 - SMOOTH_ALPHA) * prev[2] + SMOOTH_ALPHA * dip_c
                joint_curls[f] = (prev[0], prev[1], prev[2])

            smooth_thumb_cmc = (1.0 - SMOOTH_ALPHA) * smooth_thumb_cmc + SMOOTH_ALPHA * thumb_cmc_position
            smooth_thumb_mcp = (1.0 - SMOOTH_ALPHA) * smooth_thumb_mcp + SMOOTH_ALPHA * thumb_mcp_curl
            smooth_thumb_ip  = (1.0 - SMOOTH_ALPHA) * smooth_thumb_ip  + SMOOTH_ALPHA * thumb_ip_curl

            smooth_pinky_cmc = (1.0 - SMOOTH_ALPHA) * smooth_pinky_cmc + SMOOTH_ALPHA * pinky_cmc_position

            # 3) Quest → Tesollo 목표각(desired) 생성
            with state_lock:
                desired = {m: prev_target[m] for m in range(1, 21)}

                # Thumb (M1~M4)
                thumb_abad_deg = float(np.clip(
                    SPLAY_GAIN_THUMB * splay["finger1"],
                    -SPLAY_LIMIT_THUMB_DEG, SPLAY_LIMIT_THUMB_DEG
                ))
                thumb_cmc_deg = curl_to_flex_deg(smooth_thumb_cmc, FLEX_DEG_THUMB_CMC)
                thumb_mcp_deg = curl_to_flex_deg(smooth_thumb_mcp, FLEX_DEG_THUMB_MCP)
                thumb_ip_deg  = curl_to_flex_deg(smooth_thumb_ip,  FLEX_DEG_THUMB_IP)

                desired[1] = clamp_target_0p1deg(1, TARGET_SIGN[1] * int(thumb_abad_deg * 10))
                desired[2] = clamp_target_0p1deg(2, TARGET_SIGN[2] * int(thumb_cmc_deg  * 10))
                desired[3] = clamp_target_0p1deg(3, TARGET_SIGN[3] * int(thumb_mcp_deg  * 10))
                desired[4] = clamp_target_0p1deg(4, TARGET_SIGN[4] * int(thumb_ip_deg   * 10))

                # Index/Middle/Ring
                for finger_name, base_motor in [
                    ("finger2", 5),
                    ("finger3", 9),
                    ("finger4", 13),
                ]:
                    spread_deg = float(np.clip(
                        SPLAY_GAIN_DEFAULT * splay[finger_name],
                        -SPLAY_LIMIT_DEFAULT_DEG, SPLAY_LIMIT_DEFAULT_DEG
                    ))
                    mcp_c, pip_c, dip_c = joint_curls[finger_name]

                    mcp_deg = curl_to_flex_deg(mcp_c, FLEX_DEG_DEFAULT)
                    pip_deg = curl_to_flex_deg(pip_c, FLEX_DEG_DEFAULT)
                    dip_deg = curl_to_flex_deg(dip_c, FLEX_DEG_DEFAULT)

                    desired[base_motor + 0] = clamp_target_0p1deg(base_motor + 0, TARGET_SIGN[base_motor + 0] * int(spread_deg * 10))
                    desired[base_motor + 1] = clamp_target_0p1deg(base_motor + 1, TARGET_SIGN[base_motor + 1] * int(mcp_deg   * 10))
                    desired[base_motor + 2] = clamp_target_0p1deg(base_motor + 2, TARGET_SIGN[base_motor + 2] * int(pip_deg   * 10))
                    desired[base_motor + 3] = clamp_target_0p1deg(base_motor + 3, TARGET_SIGN[base_motor + 3] * int(dip_deg   * 10))

                # Pinky (M17~M20)
                pinky_spread_deg = float(np.clip(
                    SPLAY_GAIN_DEFAULT * splay["finger5"],
                    -SPLAY_LIMIT_DEFAULT_DEG, SPLAY_LIMIT_DEFAULT_DEG
                ))
                p_mcp, p_pip, p_dip = joint_curls["finger5"]

                pinky_cmc_deg = curl_to_flex_deg(smooth_pinky_cmc, 35.0)
                pinky_mcp_deg = curl_to_flex_deg(p_mcp, FLEX_DEG_DEFAULT)
                pinky_pip_deg = curl_to_flex_deg(0.7 * p_pip + 0.3 * p_dip, FLEX_DEG_DEFAULT)

                desired[17] = clamp_target_0p1deg(17, TARGET_SIGN[17] * int(pinky_cmc_deg   * 10))
                desired[18] = clamp_target_0p1deg(18, TARGET_SIGN[18] * int(pinky_spread_deg * 10))
                desired[19] = clamp_target_0p1deg(19, TARGET_SIGN[19] * int(pinky_mcp_deg   * 10))
                desired[20] = clamp_target_0p1deg(20, TARGET_SIGN[20] * int(pinky_pip_deg   * 10))

                # 4) 제어 파이프라인
                for m in range(1, 21):
                    desired[m] = clamp_step_to_current(m, desired[m], cur_pos.get(m, 0))

                enforce_motor_enable_mask(cur_pos, desired=desired, prev_target=prev_target, prev_duty=prev_duty)

                target = {}
                for m in range(1, 21):
                    limited = rate_limit_target(m, desired[m], prev_target[m])
                    target[m] = clamp_target_0p1deg(m, limited)
                    prev_target[m] = target[m]
                last_target = dict(target)

                enforce_motor_enable_mask(cur_pos, target=target, prev_target=prev_target, prev_duty=prev_duty)

                raw = {m: to_duty(target[m] - cur_pos.get(m, 0), m) for m in range(1, 21)}
                enforce_motor_enable_mask(cur_pos, raw=raw)
                raw = apply_global_limits(raw)
                enforce_motor_enable_mask(cur_pos, raw=raw)

                duty = {m: slew_limit_duty(m, raw.get(m, 0), prev_duty) for m in range(1, 21)}
                enforce_motor_enable_mask(cur_pos, duty=duty, prev_duty=prev_duty)

            # 5) Tesollo 제어 명령 전송
            try:
                gr.set_duty(duty)
            except Exception as e:
                print(f"\n[WARN] 모터 제어 실패: {e}")
                try:
                    gr.set_duty(make_zero_duty())
                except:
                    pass
                with state_lock:
                    reset_targets_to_current(cur_pos)
                    reset_duty_state()
                continue

            # 6) 상태 출력 + GUI 업데이트
            active = sum(1 for v in duty.values() if v != 0)
            total  = sum(abs(v) for v in duty.values())

            # GUI 업데이트
            visualizer.update_data(
                thumb={
                    "cmc": float(np.clip(smooth_thumb_cmc, 0.0, 1.0)),
                    "mcp": float(np.clip(smooth_thumb_mcp, 0.0, 1.0)),
                    "ip":  float(np.clip(smooth_thumb_ip,  0.0, 1.0)),
                },
                fingers={
                    "finger2": [float(np.clip(v, 0.0, 1.0)) for v in joint_curls["finger2"]],
                    "finger3": [float(np.clip(v, 0.0, 1.0)) for v in joint_curls["finger3"]],
                    "finger4": [float(np.clip(v, 0.0, 1.0)) for v in joint_curls["finger4"]],
                    "finger5": [float(np.clip(v, 0.0, 1.0)) for v in joint_curls["finger5"]],
                },
                splay=dict(splay),
                pinch=float(pinch),
                status="추적중",
                active=active,
                duty=total,
            )

            # 콘솔 출력
            idx_mcp, idx_pip, idx_dip = joint_curls["finger2"]
            pnk_mcp, pnk_pip, pnk_dip = joint_curls["finger5"]
            print(
                f"\r[CTRL] pinch={pinch:.2f} "
                f"T(c/m/i)={smooth_thumb_cmc:.2f}/{smooth_thumb_mcp:.2f}/{smooth_thumb_ip:.2f} | "
                f"Idx(m/p/d)={idx_mcp:.2f}/{idx_pip:.2f}/{idx_dip:.2f} | "
                f"Pnk(m/p/d)={pnk_mcp:.2f}/{pnk_pip:.2f}/{pnk_dip:.2f} | "
                f"active={active} duty={total}    ",
                end="", flush=True
            )

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
        visualizer.stop()
        print("[OK] 정상 종료")


if __name__ == "__main__":
    main()
