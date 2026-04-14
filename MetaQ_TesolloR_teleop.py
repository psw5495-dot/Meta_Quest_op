#!/usr/bin/env python3
"""Quest 3 → Tesollo DG-5F-M Teleoperation (최적화 압축 버전)"""
import os, time, math, socket, struct, threading, sys
import numpy as np
import tkinter as tk
from tkinter import ttk

if os.name == "nt": import msvcrt
else: import select, termios, tty

# ═══════════════════════════════════════════════════════════════════════════════
# 설정 통합
# ═══════════════════════════════════════════════════════════════════════════════
CFG = {
    'gripper_ip': "169.254.186.72", 'gripper_port': 502, 'tcp_port': 7000,
    'hz': 50, 'dt': 1.0/50, 'smooth_alpha': 0.35,
    'deadband': 8, 'max_duty_step': 40, 'duty_budget': 1500,
    'max_active': 12, 'min_duty': 18, 'protected': {17,18}
}

MOTOR_CFG = {
    1:(1.3,300,"th_opp"), 2:(1.2,320,"th_cmc"), 3:(1.0,280,"th_mcp"), 4:(0.8,200,"th_ip"),
    5:(1.2,250,"idx_sp"), 6:(0.8,200,"idx_f1"), 7:(0.8,200,"idx_f2"), 8:(0.8,200,"idx_f3"),
    9:(1.2,250,"mid_sp"), 10:(0.8,200,"mid_f1"), 11:(0.8,200,"mid_f2"), 12:(0.8,200,"mid_f3"),
    13:(1.2,250,"rng_sp"), 14:(0.8,200,"rng_f1"), 15:(0.8,200,"rng_f2"), 16:(0.8,200,"rng_f3"),
    17:(1.5,320,"pnk_sp"), 18:(1.1,260,"pnk_f1"), 19:(0.8,200,"pnk_f2"), 20:(0.8,200,"pnk_f3")
}

FINGERS = ["finger1","finger2","finger3","finger4","finger5"]
JOINT_MAP = {f"finger{i}": [i*4-3+j for j in range(4)] for i in range(1,6)}
TARGET_SIGN = {i: (-1 if i in [3,4,17,18] else 1) for i in range(1,21)}
MOTOR_EN = {m: True for m in range(1,21)}

FLEX = {'default':90.0, 'cmc':85.0, 'mcp':100.0, 'ip':90.0}
SPLAY = {'gain':1.0, 'limit':25.0, 'tgain':2.0, 'tlimit':90.0}
M2 = {'dmin':0.08, 'dmax':0.28, 'wd':0.6, 'wa':0.4}

# ═══════════════════════════════════════════════════════════════════════════════
# 수학 함수 압축
# ═══════════════════════════════════════════════════════════════════════════════
def ang_deg(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    return 180.0 if n1<1e-9 or n2<1e-9 else math.degrees(math.acos(np.clip(np.dot(v1,v2)/(n1*n2), -1, 1)))

def curl_ratio(a, open_deg=170.0, closed_deg=70.0):
    return float(np.clip((open_deg-a)/(open_deg-closed_deg), 0.0, 1.0))

def calc_joint_curls(lms, finger):
    if finger == "finger1":
        p1,p2,p3,p4 = lms[1],lms[2],lms[3],lms[4]
        return curl_ratio(ang_deg(p1-p2, p3-p2)), curl_ratio(ang_deg(p2-p3, p4-p3))
    
    idx_map = {"finger2":[5,6,7,8], "finger3":[9,10,11,12], 
               "finger4":[13,14,15,16], "finger5":[17,18,19,20]}
    mcp,pip,dip,tip = [lms[i] for i in idx_map[finger]]
    wrist = lms[0]
    return (curl_ratio(ang_deg(wrist-mcp, pip-mcp), 165.0, 70.0),
            curl_ratio(ang_deg(mcp-pip, dip-pip)),
            curl_ratio(ang_deg(pip-dip, tip-dip), 170.0, 80.0))

def calc_thumb_cmc(lms):
    td = np.linalg.norm(lms[4]-lms[0])
    hs = max(np.linalg.norm(lms[5]-lms[0]), 1e-6)
    dr = float(np.clip((np.clip(td/hs, M2['dmin'], M2['dmax']) - M2['dmin']) / 
                       (M2['dmax']-M2['dmin']), 0.0, 1.0))
    ar = curl_ratio(ang_deg(lms[0]-lms[1], lms[2]-lms[1]), 155.0, 95.0)
    return float(np.clip(M2['wd']*dr + M2['wa']*ar, 0.0, 1.0))

def calc_splay(lms):
    def dir2d(a, b): 
        v = lms[b]-lms[a]; return np.array([v[0],v[1]], dtype=np.float32)
    def signed_ang(a, b):
        a, b = a/(np.linalg.norm(a)+1e-9), b/(np.linalg.norm(b)+1e-9)
        return math.degrees(math.atan2(a[0]*b[1]-a[1]*b[0], a[0]*b[0]+a[1]*b[1]))
    
    dirs = {f"finger{i}": dir2d(*[(2,4),(5,8),(9,12),(13,16),(17,20)][i-1]) 
            for i in range(1,6)}
    base = dirs["finger3"]
    return {"finger3": 0.0, **{f: signed_ang(base, dirs[f]) for f in dirs if f != "finger3"}}

def calc_pinky_cmc(splay_val):
    return float(np.clip((np.clip(0.6*splay_val, -20.0, 20.0)+20.0)/40.0, 0.0, 1.0))

# ═══════════════════════════════════════════════════════════════════════════════
# 모터 제어 유틸리티 압축
# ═══════════════════════════════════════════════════════════════════════════════
def zero_duty(): return {m:0 for m in range(1,21)}

def toggle_motor(mid, cur, pt, pd):
    if not (1 <= mid <= 20): return f"Invalid: {mid}"
    MOTOR_EN[mid] = not MOTOR_EN[mid]
    pt[mid] = int(cur.get(mid, pt.get(mid,0))); pd[mid] = 0
    return f"M{mid:02d}({MOTOR_CFG[mid][2]}) -> {'ON' if MOTOR_EN[mid] else 'OFF'}"

def toggle_all(cur, pt, pd):
    new_state = not all(MOTOR_EN[m] for m in range(1,21))
    for m in range(1,21):
        MOTOR_EN[m] = new_state; pt[m] = int(cur.get(m,pt.get(m,0))); pd[m] = 0
    return f"All motors -> {'ON' if new_state else 'OFF'}"

def enforce_mask(cur, **kwargs):
    for m in range(1,21):
        if MOTOR_EN[m]: continue
        hold = int(cur.get(m,0))
        for k,d in kwargs.items():
            if d is not None: d[m] = hold if k in ('desired','target') else 0

def clamp_pos(mid, val):
    lims = {1:(-150,290), 2:(-850,900), 3:(-1500,290), 4:(-900,900), 
            5:(-200,310), 6:(0,1150), 17:(-300,0), 18:(-900,150)}
    lo, hi = lims.get(mid, (-900,1150))
    return int(np.clip(int(val), lo, hi))

def clamp_step(mid, desired, current):
    max_step = {2:130, 3:150, 4:200}.get(mid, 100 if mid in [1,5,9,13,17] else 250)
    return int(np.clip(int(desired), int(current)-max_step, int(current)+max_step))

def rate_limit(mid, desired, prev):
    max_spd = {2:65.0, 3:70.0, 4:90.0}.get(mid, 80.0 if mid in [1,5,9,13,17] else 100.0)
    max_delta = max(1, int(max_spd*10.0*CFG['dt']))
    return int(prev) + int(np.clip(int(desired)-int(prev), -max_delta, max_delta))

def slew_duty(mid, new_duty, prev_duty):
    prev = int(prev_duty.get(mid,0))
    limited = int(np.clip(int(new_duty), prev-CFG['max_duty_step'], prev+CFG['max_duty_step']))
    prev_duty[mid] = limited; return limited

def apply_limits(raw_duty):
    duty = {m: (v if abs(v)>=CFG['min_duty'] else 0) for m,v in raw_duty.items()}
    active = sorted([(m,abs(v)) for m,v in duty.items() if v!=0], 
                   key=lambda x:x[1], reverse=True)
    if len(active) > CFG['max_active']:
        keep = set(m for m in CFG['protected'] if duty.get(m,0)!=0)
        for m,_ in active:
            if len(keep) >= CFG['max_active']: break
            keep.add(m)
        duty = {m: (v if m in keep else 0) for m,v in duty.items()}
    
    total = sum(abs(v) for v in duty.values())
    if total > CFG['duty_budget'] and total > 0:
        scale = CFG['duty_budget'] / total
        duty = {m: int(v*scale) for m,v in duty.items()}
    return duty

def to_duty(err, mid):
    if abs(err) < CFG['deadband']: return 0
    kp, lim, _ = MOTOR_CFG[mid]
    return int(np.clip(int(kp*err), -lim, lim))

def c2flex(curl, flex_deg): return float(np.clip(curl,0.0,1.0)) * flex_deg

# ═══════════════════════════════════════════════════════════════════════════════
# 네트워크 클래스 압축
# ═══════════════════════════════════════════════════════════════════════════════
class QuestReceiver:
    def __init__(self):
        self.landmarks, self.pinch, self.connected, self.running = None, 0.0, False, True
        self.lock = threading.Lock()
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", CFG['tcp_port']))
        self.sock.listen(1); self.sock.settimeout(1.0)
        
        threading.Thread(target=self._loop, daemon=True).start()
        print(f"[TCP] Quest 3 수신 서버 시작: 포트 {CFG['tcp_port']}")

    def _loop(self):
        while self.running:
            try:
                conn, addr = self.sock.accept()
                self.connected = True; print(f"[TCP] 연결됨: {addr}")
                self._handle_client(conn)
                self.connected = False; print("[TCP] 연결 종료")
            except socket.timeout: continue
            except Exception as e: print(f"[TCP] 오류: {e}"); time.sleep(1.0)

    def _handle_client(self, conn):
        conn.settimeout(2.0); buffer = ""
        try:
            while self.running:
                data = conn.recv(4096).decode('utf-8', errors='ignore')
                if not data: break
                buffer += data
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    self._parse(line.strip())
        except: pass
        finally: conn.close()

    def _parse(self, line):
        try:
            if not line.startswith("HAND_DATA,"): return
            vals = list(map(float, line[10:].split(',')))
            if len(vals) < 64: return
            lms = np.array([[vals[i*3],vals[i*3+1],vals[i*3+2]] for i in range(21)], dtype=np.float32)
            with self.lock: self.landmarks, self.pinch = lms, vals[63]
        except: pass

    def get_data(self):
        with self.lock: return self.landmarks, self.pinch

    def stop(self):
        self.running = False
        try: self.sock.close()
        except: pass

class HandVisualizer:
    def __init__(self):
        self.lock, self.running = threading.Lock(), True
        self.data = {'thumb':{'cmc':0,'mcp':0,'ip':0}, 'fingers':{f:[0,0,0] for f in FINGERS[1:]},
                     'splay':{f:0 for f in FINGERS}, 'pinch':0, 'status':'대기', 'active':0, 'duty':0}
        threading.Thread(target=self._run_gui, daemon=True).start()

    def update(self, **kw):
        with self.lock: self.data.update(kw)

    def _run_gui(self):
        root = tk.Tk(); root.title("Hand Angles"); root.geometry("620x580")
        root.configure(bg="#2b2b2b")
        
        # 상태바
        status_frame = tk.Frame(root, bg="#313244", pady=8)
        status_frame.pack(fill="x", padx=5, pady=5)
        
        self.lbl_status = tk.Label(status_frame, text="● 대기", fg="#ff6b6b", 
                                  bg="#313244", font=("Consolas",12,"bold"))
        self.lbl_status.pack(side="left", padx=15)
        
        self.lbl_pinch = tk.Label(status_frame, text="Pinch: 0.00", fg="#4ecdc4",
                                 bg="#313244", font=("Consolas",12))
        self.lbl_pinch.pack(side="left", padx=15)
        
        self.lbl_active = tk.Label(status_frame, text="Active:0 Duty:0", fg="#fab387",
                                  bg="#313244", font=("Consolas",12))
        self.lbl_active.pack(side="right", padx=15)
        
        # 스크롤 영역
        canvas = tk.Canvas(root, bg="#2b2b2b")
        scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg="#2b2b2b")
        
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.create_window((0,0), window=scroll_frame, anchor="nw")
        canvas.pack(side="left", fill="both", expand=True, padx=(10,0))
        scrollbar.pack(side="right", fill="y", padx=(0,10))
        
        # 위젯 생성
        self.bars, self.labels = {}, {}
        finger_names = ["Thumb","Index","Middle","Ring","Pinky"]
        colors = ["#ff6b6b","#4ecdc4","#45b7d1","#96ceb4","#ffeaa7"]
        
        for i, (fname, color) in enumerate(zip(finger_names, colors)):
            fkey = FINGERS[i]
            
            # 손가락 제목
            title_frame = tk.Frame(scroll_frame, bg="#2b2b2b")
            title_frame.pack(fill="x", pady=(10 if i>0 else 0, 5))
            tk.Label(title_frame, text=f"▶ {fname}", fg=color, bg="#2b2b2b",
                    font=("Consolas",11,"bold")).pack(side="left")
            
            # 관절 바들
            joints = [("splay","Splay")] + ([("cmc","CMC"),("mcp","MCP"),("ip","IP")] 
                     if i==0 else [("mcp","MCP"),("pip","PIP"),("dip","DIP")])
            
            for jkey, jname in joints:
                joint_frame = tk.Frame(scroll_frame, bg="#2b2b2b")
                joint_frame.pack(fill="x", padx=20 if jkey=="splay" else 30, pady=1)
                
                tk.Label(joint_frame, text=f"{jname}:", fg="#999", bg="#2b2b2b",
                        font=("Consolas",9), width=8, anchor="w").pack(side="left")
                
                bar = ttk.Progressbar(joint_frame, length=220, mode='determinate')
                bar.pack(side="left", padx=5)
                
                lbl = tk.Label(joint_frame, text="0.00", fg="#ccc", bg="#2b2b2b",
                              font=("Consolas",9), width=8)
                lbl.pack(side="left", padx=5)
                
                self.bars[(fkey,jkey)] = bar
                self.labels[(fkey,jkey)] = lbl
        
        self._schedule_update(root)
        try: root.mainloop()
        except: pass

    def _schedule_update(self, root):
        if not self.running: return
        try: self._update_display()
        except: pass
        root.after(33, lambda: self._schedule_update(root))

    def _update_display(self):
        with self.lock: data = dict(self.data)
        
        pinch, status = data['pinch'], data['status']
        if pinch > 0.5: self.lbl_status.config(text="● PINCH", fg="#a6e3a1")
        elif status == "추적": self.lbl_status.config(text="● 추적중", fg="#4ecdc4")
        else: self.lbl_status.config(text="● 대기", fg="#ff6b6b")
        
        self.lbl_pinch.config(text=f"Pinch: {pinch:.2f}")
        self.lbl_active.config(text=f"Active:{data['active']} Duty:{data['duty']}")
        
        # 엄지
        for joint in ["cmc","mcp","ip"]:
            val = data['thumb'].get(joint, 0.0)
            key = ("finger1", joint)
            if key in self.bars:
                self.bars[key]['value'] = val*100
                self.labels[key].config(text=f"{val:.2f}")
        
        # 다른 손가락들
        for i, fkey in enumerate(FINGERS[1:], 1):
            vals = data['fingers'].get(fkey, [0,0,0])
            for j, joint in enumerate(["mcp","pip","dip"]):
                key = (fkey, joint)
                if key in self.bars:
                    self.bars[key]['value'] = vals[j]*100
                    self.labels[key].config(text=f"{vals[j]:.2f}")
        
        # Splay
        for fkey in FINGERS:
            sval = data['splay'].get(fkey, 0.0)
            key = (fkey, "splay")
            if key in self.bars:
                norm_val = max(0, min(100, (sval+30.0)/60.0*100))
                self.bars[key]['value'] = norm_val
                self.labels[key].config(text=f"{sval:+.1f}°")

    def stop(self):
        self.running = False

class TesolloClient:
    def __init__(self):
        self.sock = None
    
    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(0.5)
        self.sock.connect((CFG['gripper_ip'], CFG['gripper_port']))
        print(f"[OK] Tesollo 연결: {CFG['gripper_ip']}:{CFG['gripper_port']}")
    
    def _recv_exact(self, n):
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk: raise ConnectionError("Socket closed")
            buf += chunk
        return buf
    
    def get_positions(self):
        pkt = struct.pack(">HBB", 3, 1, 1)
        self.sock.sendall(pkt)
        resp_len = struct.unpack(">H", self._recv_exact(2))[0]
        resp = self._recv_exact(resp_len - 2)
        if not resp or resp[0] != 1: raise RuntimeError("Bad response")
        
        pos, i = {}, 1
        while i + 2 < len(resp):
            pos[resp[i]] = struct.unpack(">h", resp[i+1:i+3])[0]
            i += 3
        return pos
    
    def set_duty(self, duty_dict):
        data = b"".join(struct.pack("Bh", jid, int(np.clip(duty_dict.get(jid,0), -1000, 1000)))
                       for jid in range(1, 21))
        pkt = struct.pack(">HB", len(data)+3, 5) + data
        self.sock.sendall(pkt)
    
    def close(self):
        if self.sock:
            try: self.sock.close()
            except: pass

# ═══════════════════════════════════════════════════════════════════════════════
# 메인 함수
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("═"*60)
    print("Quest 3 → Tesollo DG-5F-M Teleoperation (최적화 버전)")
    print(f"TCP:{CFG['tcp_port']} Tesollo:{CFG['gripper_ip']}:{CFG['gripper_port']} Ctrl+C 종료")
    print("═"*60)

    receiver = QuestReceiver()
    visualizer = HandVisualizer()
    gripper = TesolloClient()
    gripper.connect()

    # 상태 초기화
    smooth_joints = {f: [0.0,0.0,0.0] for f in FINGERS[1:]}
    smooth_splay = {f: 0.0 for f in FINGERS}
    smooth_thumb = {'cmc':0.0, 'mcp':0.0, 'ip':0.0}
    smooth_pinky_cmc = 0.0
    
    prev_target = {m:0 for m in range(1,21)}
    prev_duty = {m:0 for m in range(1,21)}
    cur_pos = {m:0 for m in range(1,21)}
    target_valid = False
    
    state_lock = threading.Lock()

    def reset_to_current():
        nonlocal target_valid
        for m in range(1,21): prev_target[m] = int(cur_pos.get(m,0))
        target_valid = True

    def reset_duty():
        for m in range(1,21): prev_duty[m] = 0

    # 키보드 입력 스레드
    def keyboard_thread():
        nonlocal cur_pos, prev_target, prev_duty
        print("\n[KEY] T:전체토글 | 숫자:개별모터 | Q:종료\n")
        
        def exec_toggle(mid):
            with state_lock: msg = toggle_motor(mid, cur_pos, prev_target, prev_duty)
            print(f"\n[KEY] {msg}")
        
        def exec_all():
            with state_lock: msg = toggle_all(cur_pos, prev_target, prev_duty)
            print(f"\n[KEY] {msg}")

        if os.name == "nt":
            while True:
                if msvcrt.kbhit():
                    ch = msvcrt.getwch().upper()
                    if ch == 'T': exec_all()
                    elif ch == 'Q': break
                    elif ch.isdigit() and 1 <= int(ch) <= 9: exec_toggle(int(ch))
                time.sleep(0.01)
        else:
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                while True:
                    if select.select([sys.stdin],[],[],0.01)[0]:
                        ch = sys.stdin.read(1).upper()
                        if ch == 'T': exec_all()
                        elif ch == 'Q': break
                        elif ch.isdigit() and 1 <= int(ch) <= 9: exec_toggle(int(ch))
                    time.sleep(0.005)
            finally: termios.tcsetattr(fd, termios.TCSADRAIN, old)

    threading.Thread(target=keyboard_thread, daemon=True).start()

    # 메인 루프
    no_hand_frames, last_time = 0, time.time()
    
    try:
        while True:
            # 주기 제어
            now = time.time()
            if now - last_time < CFG['dt']: time.sleep(CFG['dt'] - (now - last_time))
            last_time = time.time()

            # Tesollo 위치 읽기
            try:
                new_pos = gripper.get_positions()
                with state_lock: cur_pos = new_pos
            except Exception as e:
                print(f"\n[WARN] Tesollo 오류: {e}")
                try: gripper.set_duty(zero_duty())
                except: pass
                with state_lock: reset_duty(); target_valid = False
                continue

            with state_lock:
                if not target_valid: reset_to_current(); reset_duty()

            # Quest 손 데이터
            landmarks, pinch = receiver.get_data()

            if landmarks is None:
                no_hand_frames += 1
                if no_hand_frames >= 10:
                    try: gripper.set_duty(zero_duty())
                    except: pass
                    with state_lock: reset_to_current(); reset_duty()
                
                status = "연결대기" if not receiver.connected else "손 미감지"
                visualizer.update(status=status, pinch=0.0)
                print(f"\r[INFO] {status} ({no_hand_frames}f)    ", end="", flush=True)
                continue

            no_hand_frames = 0

            # 손 특성 계산
            splay = calc_splay(landmarks)
            thumb_cmc = calc_thumb_cmc(landmarks)
            thumb_mcp, thumb_ip = calc_thumb_cmc(landmarks)  # 엄지 MCP, IP
            
            joint_curls = {}
            for f in FINGERS[1:]:
                joint_curls[f] = calc_joint_curls(landmarks, f)
            
            pinky_cmc = calc_pinky_cmc(splay["finger5"])

            # 스무딩
            alpha = CFG['smooth_alpha']
            for f in FINGERS:
                smooth_splay[f] = (1-alpha)*smooth_splay[f] + alpha*splay[f]
            
            smooth_thumb['cmc'] = (1-alpha)*smooth_thumb['cmc'] + alpha*thumb_cmc
            smooth_thumb['mcp'] = (1-alpha)*smooth_thumb['mcp'] + alpha*thumb_mcp
            smooth_thumb['ip'] = (1-alpha)*smooth_thumb['ip'] + alpha*thumb_ip
            smooth_pinky_cmc = (1-alpha)*smooth_pinky_cmc + alpha*pinky_cmc
            
            for f in FINGERS[1:]:
                for i in range(3):
                    smooth_joints[f][i] = (1-alpha)*smooth_joints[f][i] + alpha*joint_curls[f][i]

            # 제어 계산
            with state_lock:
                desired = {m: prev_target[m] for m in range(1,21)}

                # 엄지 (M1-M4)
                thumb_splay_deg = float(np.clip(SPLAY['tgain']*smooth_splay["finger1"], 
                                              -SPLAY['tlimit'], SPLAY['tlimit']))
                desired[1] = clamp_pos(1, TARGET_SIGN[1] * int(thumb_splay_deg * 10))
                desired[2] = clamp_pos(2, TARGET_SIGN[2] * int(c2flex(smooth_thumb['cmc'], FLEX['cmc']) * 10))
                desired[3] = clamp_pos(3, TARGET_SIGN[3] * int(c2flex(smooth_thumb['mcp'], FLEX['mcp']) * 10))
                desired[4] = clamp_pos(4, TARGET_SIGN[4] * int(c2flex(smooth_thumb['ip'], FLEX['ip']) * 10))

                # 검지/중지/약지 (M5-M16)
                for fname, base_motor in [("finger2",5), ("finger3",9), ("finger4",13)]:
                    spread_deg = float(np.clip(SPLAY['gain']*smooth_splay[fname], 
                                             -SPLAY['limit'], SPLAY['limit']))
                    mcp_c, pip_c, dip_c = smooth_joints[fname]
                    
                    desired[base_motor+0] = clamp_pos(base_motor+0, TARGET_SIGN[base_motor+0] * int(spread_deg * 10))
                    desired[base_motor+1] = clamp_pos(base_motor+1, TARGET_SIGN[base_motor+1] * int(c2flex(mcp_c, FLEX['default']) * 10))
                    desired[base_motor+2] = clamp_pos(base_motor+2, TARGET_SIGN[base_motor+2] * int(c2flex(pip_c, FLEX['default']) * 10))
                    desired[base_motor+3] = clamp_pos(base_motor+3, TARGET_SIGN[base_motor+3] * int(c2flex(dip_c, FLEX['default']) * 10))

                # 새끼 (M17-M20)
                pinky_spread_deg = float(np.clip(SPLAY['gain']*smooth_splay["finger5"], 
                                               -SPLAY['limit'], SPLAY['limit']))
                p_mcp, p_pip, p_dip = smooth_joints["finger5"]
                
                desired[17] = clamp_pos(17, TARGET_SIGN[17] * int(c2flex(smooth_pinky_cmc, 35.0) * 10))
                desired[18] = clamp_pos(18, TARGET_SIGN[18] * int(pinky_spread_deg * 10))
                desired[19] = clamp_pos(19, TARGET_SIGN[19] * int(c2flex(p_mcp, FLEX['default']) * 10))
                desired[20] = clamp_pos(20, TARGET_SIGN[20] * int(c2flex(0.7*p_pip+0.3*p_dip, FLEX['default']) * 10))

                # 제어 파이프라인
                for m in range(1,21):
                    desired[m] = clamp_step(m, desired[m], cur_pos.get(m,0))
                
                enforce_mask(cur_pos, desired=desired, prev_target=prev_target, prev_duty=prev_duty)

                target = {}
                for m in range(1,21):
                    target[m] = clamp_pos(m, rate_limit(m, desired[m], prev_target[m]))
                    prev_target[m] = target[m]
                
                enforce_mask(cur_pos, target=target, prev_target=prev_target, prev_duty=prev_duty)

                raw_duty = {m: to_duty(target[m] - cur_pos.get(m,0), m) for m in range(1,21)}
                enforce_mask(cur_pos, raw=raw_duty)
                
                limited_duty = apply_limits(raw_duty)
                enforce_mask(cur_pos, raw=limited_duty)

                final_duty = {m: slew_duty(m, limited_duty.get(m,0), prev_duty) for m in range(1,21)}
                enforce_mask(cur_pos, duty=final_duty, prev_duty=prev_duty)

            # Tesollo 명령 전송
            try: gripper.set_duty(final_duty)
            except Exception as e:
                print(f"\n[WARN] 모터 제어 실패: {e}")
                try: gripper.set_duty(zero_duty())
                except: pass
                with state_lock: reset_to_current(); reset_duty()
                continue

            # GUI 업데이트
            active = sum(1 for v in final_duty.values() if v != 0)
            total = sum(abs(v) for v in final_duty.values())
            
            visualizer.update(
                thumb={'cmc':float(np.clip(smooth_thumb['cmc'],0,1)),
                       'mcp':float(np.clip(smooth_thumb['mcp'],0,1)),
                       'ip':float(np.clip(smooth_thumb['ip'],0,1))},
                fingers={f: [float(np.clip(v,0,1)) for v in smooth_joints[f]] 
                        for f in FINGERS[1:]},
                splay=dict(smooth_splay),
                pinch=float(pinch),
                status="추적",
                active=active,
                duty=total
            )

            # 콘솔 출력
            print(f"\r[CTRL] pinch={pinch:.2f} active={active} duty={total}    ", 
                  end="", flush=True)

    except KeyboardInterrupt:
        print("\n[INFO] 종료 중...")
    finally:
        try: gripper.set_duty(zero_duty()); print("[OK] 모터 정지")
        except: pass
        gripper.close(); receiver.stop(); visualizer.stop()
        print("[OK] 정상 종료")

if __name__ == "__main__":
    main()
