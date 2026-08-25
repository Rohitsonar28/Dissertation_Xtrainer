import os
import sys
import time
import socket
import numpy as np

# Dynamixel SDK Path Inclusion
SDK_PATH = r"C:\Users\Rohit\Downloads\dobot_xtrainer_0813\dobot_xtrainer-master"
if SDK_PATH not in sys.path:
    sys.path.insert(0, SDK_PATH)

from dynamixel_sdk import PortHandler, PacketHandler

os.system("")  # Enable ANSI formatting on Windows


class DualArmBimanualPrecisionTeleop:
    def __init__(
        self,
        left_robot_ip="192.168.5.1",
        right_robot_ip="192.168.5.2",
        left_master_port="COM5",
        right_master_port="COM3",
        left_gripper_port="COM6",
        right_gripper_port="COM7",
        master_baudrate=2000000,
        gripper_baudrate=1000000,
    ):
        self.left_robot_ip = left_robot_ip
        self.right_robot_ip = right_robot_ip
        self.left_master_port = left_master_port
        self.right_master_port = right_master_port
        self.left_gripper_port = left_gripper_port
        self.right_gripper_port = right_gripper_port
        self.master_baudrate = int(master_baudrate)
        self.gripper_baudrate = int(gripper_baudrate)

        # ------------------------------------------------------------
        # TARGET HOME POSES (Degrees)
        # ------------------------------------------------------------
        self.left_home_pose = [-89.7758, 1.3375, -86.5728, -2.3068, 87.0675, -0.4531]
        self.right_home_pose = [89.7758, 1.3375, 86.5728, 2.3068, -87.0675, 0.4531]

        # ------------------------------------------------------------
        # MASTER HARDWARE MAPPING
        # ------------------------------------------------------------
        # Left Master (COM5)
        self.left_joint_ids = [1, 2, 4, 5, 6, 7]
        self.left_all_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        self.left_arm_ids = [1, 2, 3, 4, 5, 6, 7]
        self.left_trigger_id = 8
        self.left_joint_signs = np.array([1, 1, -1, -1, -1, 1], dtype=np.float64)

        # Right Master (COM3)
        self.right_joint_ids = [11, 12, 14, 15, 16, 17]
        self.right_all_ids = [11, 12, 13, 14, 15, 16, 17, 18]
        self.right_arm_ids = [11, 12, 13, 14, 15, 16, 17]
        self.right_trigger_id = 18
        self.right_joint_signs = np.array([1, 1, -1, -1, -1, 1], dtype=np.float64)

        # Dynamixel Protocol 2.0 Registers
        self.ADDR_OPERATING_MODE = 11
        self.ADDR_TORQUE_ENABLE = 64
        self.ADDR_LED = 65
        self.ADDR_GOAL_CURRENT = 102
        self.ADDR_GOAL_POSITION = 116
        self.ADDR_PRESENT_POSITION = 132

        # ------------------------------------------------------------
        # ARM MOTION DYNAMICS (50 Hz)
        # ------------------------------------------------------------
        self.ARM_ALPHA = 0.38             # Responsive trajectory tracking
        self.DEADBAND_DEG = 0.08          # Micro-jitter elimination
        self.MAX_VELOCITY_DEG = 8.5       # Step velocity clamp

        # ------------------------------------------------------------
        # GRIPPER SPECIFICATIONS & TUNING
        # ------------------------------------------------------------
        self.BROADCAST_ID = 254

        # Left Gripper (COM6 | ID 22)
        self.L_GRIPPER_ID = 22
        self.L_GRIPPER_OPEN = 2000
        self.L_GRIPPER_CLOSED = 980
        self.L_TOTAL_STROKE = self.L_GRIPPER_OPEN - self.L_GRIPPER_CLOSED  # 1020 ticks
        self.L_TORQUE_LIMIT = 850
        self.L_TRIGGER_TRAVEL = 210.0

        # Right Gripper (COM7 | ID 21)
        self.R_GRIPPER_ID = 21
        self.R_GRIPPER_OPEN = 3865
        self.R_GRIPPER_CLOSED = 2800
        self.R_TOTAL_STROKE = self.R_GRIPPER_OPEN - self.R_GRIPPER_CLOSED  # 1065 ticks
        self.R_TORQUE_LIMIT = 920
        self.R_TRIGGER_TRAVEL = 200.0     # Calibrated for full-stroke closing

        # Trigger Sensitivity Tuning
        self.PRECISION_BUCKETS = 100.0    # 1% precision steps
        self.GRIP_ALPHA = 0.78            # Fast tracking alpha
        self.TRIGGER_DEADBAND = 4.0

        # Bilateral Haptics
        self.HAPTIC_INTENSITY = 0.70
        self.LOAD_DEADBAND = 40
        self.MAX_FEEDBACK_CURRENT = 260
        self.HAPTIC_SMOOTH_ALPHA = 0.25

        # State Vectors - Left Arm
        self.left_baseline_robot = np.array(self.left_home_pose, dtype=np.float64)
        self.left_locked_ticks = None
        self.left_smoothed_joints = np.array(self.left_home_pose, dtype=np.float64)
        self.left_trigger_idle_ticks = 2048
        self.left_smoothed_squeeze_pct = 0.0
        self.left_last_sent_gripper_reg = None
        self.left_last_known_pos = self.L_GRIPPER_OPEN
        self.left_last_known_load = 0
        self.left_filtered_haptic_current = 0.0
        self.left_last_sent_current = None
        self.left_state = "RUNNING"
        self.left_last_btn_time = 0.0

        # State Vectors - Right Arm
        self.right_baseline_robot = np.array(self.right_home_pose, dtype=np.float64)
        self.right_locked_ticks = None
        self.right_smoothed_joints = np.array(self.right_home_pose, dtype=np.float64)
        self.right_trigger_idle_ticks = 1964
        self.right_smoothed_squeeze_pct = 0.0
        self.right_last_sent_gripper_reg = None
        self.right_last_known_pos = self.R_GRIPPER_OPEN
        self.right_last_known_load = 0
        self.right_filtered_haptic_current = 0.0
        self.right_last_sent_current = None
        self.right_state = "RUNNING"
        self.right_last_btn_time = 0.0

        # Hardware Handlers
        self.l_master_ph = None
        self.l_master_pkh = None
        self.r_master_ph = None
        self.r_master_pkh = None

        self.l_grip_ph = None
        self.l_grip_pkh = None
        self.r_grip_ph = None
        self.r_grip_pkh = None

        self.l_sock = None
        self.r_sock = None

    # ------------------------------------------------------------
    # LOW-LEVEL ROBOT COMMS (TCP_NODELAY + LOOKAHEAD INTERPOLATION)
    # ------------------------------------------------------------
    def _send_cmd_sync(self, sock, cmd_str):
        try:
            msg = (cmd_str + "\r\n").encode("utf-8")
            sock.sendall(msg)
            time.sleep(0.05)
            return sock.recv(1024).decode("utf-8").strip()
        except Exception as e:
            return f"Error: {e}"

    def _send_servoj_fast(self, sock, joint_angles):
        j_str = ",".join(f"{float(x):.4f}" for x in joint_angles)
        msg = f"ServoJ({j_str},t=0.03,lookahead=100,gain=300)\r\n".encode("utf-8")
        try:
            sock.sendall(msg)
            sock.setblocking(False)
            try:
                sock.recv(1024)
            except (BlockingIOError, socket.error):
                pass
            sock.setblocking(True)
        except Exception:
            pass

    # ------------------------------------------------------------
    # MASTER HARDWARE METHODS
    # ------------------------------------------------------------
    def set_master_arm_torque(self, side="both", enable=False):
        val = 1 if enable else 0
        try:
            if side in ("left", "both") and self.l_master_pkh and self.l_master_ph:
                for dxl_id in self.left_arm_ids:
                    self.l_master_pkh.write1ByteTxRx(self.l_master_ph, int(dxl_id), int(self.ADDR_TORQUE_ENABLE), val)
            if side in ("right", "both") and self.r_master_pkh and self.r_master_ph:
                for dxl_id in self.right_arm_ids:
                    self.r_master_pkh.write1ByteTxRx(self.r_master_ph, int(dxl_id), int(self.ADDR_TORQUE_ENABLE), val)
        except Exception:
            pass

    def setup_master_haptic_modes(self):
        try:
            # Left Trigger Haptics
            self.l_master_pkh.write1ByteTxRx(self.l_master_ph, int(self.left_trigger_id), int(self.ADDR_TORQUE_ENABLE), 0)
            time.sleep(0.02)
            self.l_master_pkh.write1ByteTxRx(self.l_master_ph, int(self.left_trigger_id), int(self.ADDR_OPERATING_MODE), 5)
            self.l_master_pkh.write1ByteTxRx(self.l_master_ph, int(self.left_trigger_id), int(self.ADDR_TORQUE_ENABLE), 1)
            time.sleep(0.02)

            # Right Trigger Haptics
            self.r_master_pkh.write1ByteTxRx(self.r_master_ph, int(self.right_trigger_id), int(self.ADDR_TORQUE_ENABLE), 0)
            time.sleep(0.02)
            self.r_master_pkh.write1ByteTxRx(self.r_master_ph, int(self.right_trigger_id), int(self.ADDR_OPERATING_MODE), 5)
            self.r_master_pkh.write1ByteTxRx(self.r_master_ph, int(self.right_trigger_id), int(self.ADDR_TORQUE_ENABLE), 1)
            time.sleep(0.02)
        except Exception:
            pass

    def set_leds(self, side="both", enable=True):
        val = 1 if enable else 0
        try:
            if side in ("left", "both") and self.l_master_pkh and self.l_master_ph:
                for dxl_id in self.left_all_ids:
                    self.l_master_pkh.write1ByteTxRx(self.l_master_ph, int(dxl_id), int(self.ADDR_LED), val)
            if side in ("right", "both") and self.r_master_pkh and self.r_master_ph:
                for dxl_id in self.right_all_ids:
                    self.r_master_pkh.write1ByteTxRx(self.r_master_ph, int(dxl_id), int(self.ADDR_LED), val)
        except Exception:
            pass

    def read_encoders_safe(self, pkh, ph, joint_ids, fallback_ticks=None):
        raw_ticks = []
        for idx, dxl_id in enumerate(joint_ids):
            default_val = float(fallback_ticks[idx]) if fallback_ticks is not None else 2048.0
            try:
                pos, res, _ = pkh.read4ByteTxRx(ph, int(dxl_id), int(self.ADDR_PRESENT_POSITION))
                if res != 0:
                    raw_ticks.append(default_val)
                    continue

                pos = int(pos) & 0xFFFFFFFF
                if pos > 0x7FFFFFFF:
                    pos -= 0x100000000

                if pos < -100000 or pos > 100000:
                    raw_ticks.append(default_val)
                else:
                    raw_ticks.append(float(pos))
            except Exception:
                raw_ticks.append(default_val)

        return np.array(raw_ticks, dtype=np.float64)

    def read_trigger_safe(self, pkh, ph, trigger_id, fallback_val):
        try:
            pos, res, _ = pkh.read4ByteTxRx(ph, int(trigger_id), int(self.ADDR_PRESENT_POSITION))
            if res != 0:
                return fallback_val
            pos = int(pos) & 0xFFFFFFFF
            if pos > 0x7FFFFFFF:
                pos -= 0x100000000
            return pos
        except Exception:
            return fallback_val

    def get_official_keys(self, ph):
        try:
            ph.clearPort()
            ph.writePort([0xAA, 0x55, 0xAA])
            rx = []
            t0 = time.time()
            while len(rx) < 4 and (time.time() - t0) < 0.015:
                chunk = ph.readPort(int(4 - len(rx)))
                if chunk:
                    rx.extend(chunk)

            if len(rx) >= 4 and (int(rx[0]) + int(rx[1]) + int(rx[2]) + int(rx[3])) == (255 * 2):
                key1 = (int(rx[0]) >> 4) & 0x0F
                key2 = int(rx[0]) & 0x0F
                prox = int(rx[2])
                return (key1 == 0), (key2 == 0), (prox == 0)
        except Exception:
            pass
        return False, False, False

    def apply_smooth_trigger_haptics(self, side, slave_load):
        if side == "left":
            pkh, ph, trig_id = self.l_master_pkh, self.l_master_ph, self.left_trigger_id
            idle_ticks = self.left_trigger_idle_ticks
            filtered_curr = self.left_filtered_haptic_current
            last_sent = self.left_last_sent_current
        else:
            pkh, ph, trig_id = self.r_master_pkh, self.r_master_ph, self.right_trigger_id
            idle_ticks = self.right_trigger_idle_ticks
            filtered_curr = self.right_filtered_haptic_current
            last_sent = self.right_last_sent_current

        load_mag = slave_load & 0x03FF
        if load_mag > self.LOAD_DEADBAND:
            excess_load = load_mag - self.LOAD_DEADBAND
            raw_target_current = float(excess_load * self.HAPTIC_INTENSITY)
            raw_target_current = min(float(self.MAX_FEEDBACK_CURRENT), raw_target_current)
        else:
            raw_target_current = 0.0

        filtered_curr = (
            self.HAPTIC_SMOOTH_ALPHA * raw_target_current + (1.0 - self.HAPTIC_SMOOTH_ALPHA) * filtered_curr
        )
        target_int = int(round(filtered_curr))

        if last_sent is None or abs(target_int - last_sent) >= 3:
            try:
                pkh.write4ByteTxRx(ph, int(trig_id), int(self.ADDR_GOAL_POSITION), int(idle_ticks))
                pkh.write2ByteTxRx(ph, int(trig_id), int(self.ADDR_GOAL_CURRENT), target_int)
                if side == "left":
                    self.left_last_sent_current = target_int
                else:
                    self.right_last_sent_current = target_int
            except Exception:
                pass

        if side == "left":
            self.left_filtered_haptic_current = filtered_curr
        else:
            self.right_filtered_haptic_current = filtered_curr

        return target_int

    # ------------------------------------------------------------
    # CALIBRATED 2% GRIPPER METHODS
    # ------------------------------------------------------------
    def read_gripper_pos(self, side):
        if side == "left":
            pkh, ph, g_id = self.l_grip_pkh, self.l_grip_ph, self.L_GRIPPER_ID
            last_pos = self.left_last_known_pos
        else:
            pkh, ph, g_id = self.r_grip_pkh, self.r_grip_ph, self.R_GRIPPER_ID
            last_pos = self.right_last_known_pos

        try:
            pos, res, _ = pkh.read2ByteTxRx(ph, g_id, 56)
            if res == 0 and pos is not None:
                if side == "left":
                    self.left_last_known_pos = pos
                else:
                    self.right_last_known_pos = pos
                return pos
        except Exception:
            pass
        return last_pos

    def read_gripper_load(self, side):
        if side == "left":
            pkh, ph, g_id = self.l_grip_pkh, self.l_grip_ph, self.L_GRIPPER_ID
            last_load = self.left_last_known_load
        else:
            pkh, ph, g_id = self.r_grip_pkh, self.r_grip_ph, self.R_GRIPPER_ID
            last_load = self.right_last_known_load

        try:
            load, res, _ = pkh.read2ByteTxRx(ph, g_id, 60)
            if res == 0 and load is not None:
                mag = load & 0x03FF
                if side == "left":
                    self.left_last_known_load = mag
                else:
                    self.right_last_known_load = mag
                return mag
        except Exception:
            pass
        return last_load

    def release_gripper_torque_clean(self, side="both"):
        sides = ["left", "right"] if side == "both" else [side]
        for s in sides:
            pkh = self.l_grip_pkh if s == "left" else self.r_grip_pkh
            ph = self.l_grip_ph if s == "left" else self.r_grip_ph
            g_id = self.L_GRIPPER_ID if s == "left" else self.R_GRIPPER_ID
            if not ph:
                continue
            try:
                pkh.write1ByteTxRx(ph, g_id, 55, 0)
                pkh.write1ByteTxRx(ph, g_id, 40, 0)
                pkh.write2ByteTxRx(ph, g_id, 48, 0)
            except Exception:
                pass

    def configure_microstep_gripper(self, side="both", enable=True):
        sides = ["left", "right"] if side == "both" else [side]
        for s in sides:
            pkh = self.l_grip_pkh if s == "left" else self.r_grip_pkh
            ph = self.l_grip_ph if s == "left" else self.r_grip_ph
            g_id = self.L_GRIPPER_ID if s == "left" else self.R_GRIPPER_ID
            torque_lim = self.L_TORQUE_LIMIT if s == "left" else self.R_TORQUE_LIMIT
            if not ph:
                continue
            try:
                if enable:
                    # 1. Unlock EEPROM
                    pkh.write1ByteTxRx(ph, g_id, 55, 0)
                    time.sleep(0.02)

                    # 2. Operating Mode & Angle Limits
                    pkh.write1ByteTxRx(ph, g_id, 33, 0)      # Position Control Mode
                    pkh.write2ByteTxRx(ph, g_id, 9, 0)
                    pkh.write2ByteTxRx(ph, g_id, 11, 4095)

                    # 3. Enhanced Stiff PID
                    pkh.write1ByteTxRx(ph, g_id, 21, 42)     # Stiff P Gain
                    pkh.write1ByteTxRx(ph, g_id, 22, 18)     # D Gain
                    pkh.write1ByteTxRx(ph, g_id, 23, 4)      # Active I Gain
                    pkh.write1ByteTxRx(ph, g_id, 24, 0)      # 0-Tick Deadband

                    # 4. Torque & Motion Profile Setup
                    pkh.write2ByteTxRx(ph, g_id, 48, torque_lim)
                    pkh.write2ByteTxRx(ph, g_id, 44, 3000)       # High speed execution
                    pkh.write1ByteTxRx(ph, g_id, 41, 80)         # Fast acceleration
                    pkh.write1ByteTxRx(ph, g_id, 40, 1)          # Torque ENABLE
                    time.sleep(0.04)
                else:
                    self.release_gripper_torque_clean(side=s)
            except Exception:
                pass

    def move_percentage_microstep(self, side, raw_pct, blocking=False):
        raw_pct = float(np.clip(raw_pct, 0.0, 1.0))
        quantized_pct = np.round(raw_pct * self.PRECISION_BUCKETS) / self.PRECISION_BUCKETS

        if side == "left":
            self.left_smoothed_squeeze_pct = (
                self.GRIP_ALPHA * quantized_pct + (1.0 - self.GRIP_ALPHA) * self.left_smoothed_squeeze_pct
            )
            target = int(round(self.L_GRIPPER_OPEN - self.left_smoothed_squeeze_pct * self.L_TOTAL_STROKE))
            pkh, ph, g_id = self.l_grip_pkh, self.l_grip_ph, self.L_GRIPPER_ID
            last_sent = self.left_last_sent_gripper_reg
        else:
            self.right_smoothed_squeeze_pct = (
                self.GRIP_ALPHA * quantized_pct + (1.0 - self.GRIP_ALPHA) * self.right_smoothed_squeeze_pct
            )
            target = int(round(self.R_GRIPPER_OPEN - self.right_smoothed_squeeze_pct * self.R_TOTAL_STROKE))
            pkh, ph, g_id = self.r_grip_pkh, self.r_grip_ph, self.R_GRIPPER_ID
            last_sent = self.right_last_sent_gripper_reg

        if not blocking:
            if last_sent is None or abs(target - last_sent) >= 1:
                try:
                    pkh.write2ByteTxRx(ph, g_id, 42, target)
                    if side == "left":
                        self.left_last_sent_gripper_reg = target
                    else:
                        self.right_last_sent_gripper_reg = target
                except Exception:
                    pass
            return target

        try:
            pkh.write2ByteTxRx(ph, g_id, 42, target)
            if side == "left":
                self.left_last_sent_gripper_reg = target
            else:
                self.right_last_sent_gripper_reg = target
        except Exception:
            pass

        t0 = time.time()
        while time.time() - t0 < 3.0:
            curr = self.read_gripper_pos(side)
            if curr is not None:
                sys.stdout.write(f"\r\033[K    [{side.upper()}] Moving to: {target:4d} | Current: {curr:4d} ticks")
                sys.stdout.flush()
                if abs(curr - target) <= 5:
                    break
            time.sleep(0.02)
        print()
        return target

    # ============================================================
    # STEP 1: INITIALIZE HARDWARE & MOVE BOTH ARMS
    # ============================================================
    def step1_travel_both_robots_to_coordinates(self):
        print("\n==================================================")
        print(" [STEP 1] INITIALIZING HARDWARE & MOVING ROBOT ARMS")
        print("==================================================")

        # 1. Connect Master Handles
        print(f"[+] Connecting Left Master on {self.left_master_port}...")
        self.l_master_ph = PortHandler(self.left_master_port)
        self.l_master_pkh = PacketHandler(2.0)
        if not self.l_master_ph.openPort() or not self.l_master_ph.setBaudRate(self.master_baudrate):
            raise RuntimeError(f"Could not open {self.left_master_port}")

        print(f"[+] Connecting Right Master on {self.right_master_port}...")
        self.r_master_ph = PortHandler(self.right_master_port)
        self.r_master_pkh = PacketHandler(2.0)
        if not self.r_master_ph.openPort() or not self.r_master_ph.setBaudRate(self.master_baudrate):
            raise RuntimeError(f"Could not open {self.right_master_port}")

        self.set_master_arm_torque("both", enable=False)
        self.set_leds("both", enable=False)

        # 2. Connect Gripper COM Ports
        print(f"[+] Connecting Left Gripper on {self.left_gripper_port}...")
        self.l_grip_ph = PortHandler(self.left_gripper_port)
        self.l_grip_pkh = PacketHandler(1.0)
        if not self.l_grip_ph.openPort() or not self.l_grip_ph.setBaudRate(self.gripper_baudrate):
            raise RuntimeError(f"Could not open {self.left_gripper_port}")

        print(f"[+] Connecting Right Gripper on {self.right_gripper_port}...")
        self.r_grip_ph = PortHandler(self.right_gripper_port)
        self.r_grip_pkh = PacketHandler(1.0)
        if not self.r_grip_ph.openPort() or not self.r_grip_ph.setBaudRate(self.gripper_baudrate):
            raise RuntimeError(f"Could not open {self.right_gripper_port}")

        # 3. Connect Robot Sockets with Zero-Latency TCP_NODELAY
        print(f"[+] Connecting Left Nova 2 at {self.left_robot_ip}:29999...")
        self.l_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.l_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.l_sock.settimeout(5.0)
        self.l_sock.connect((self.left_robot_ip, 29999))

        print(f"[+] Connecting Right Nova 2 at {self.right_robot_ip}:29999...")
        self.r_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.r_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.r_sock.settimeout(5.0)
        self.r_sock.connect((self.right_robot_ip, 29999))

        for name, sock in [("Left", self.l_sock), ("Right", self.r_sock)]:
            print(f"[+] Initializing {name} Robot...")
            self._send_cmd_sync(sock, "ClearError()")
            time.sleep(0.5)
            self._send_cmd_sync(sock, "PowerOn()")
            time.sleep(3.0)
            self._send_cmd_sync(sock, "EnableRobot()")
            time.sleep(1.5)
            mode_str = self._send_cmd_sync(sock, "RobotMode()")
            if "{5}" not in mode_str:
                self._send_cmd_sync(sock, "EnableRobot(0)")
                time.sleep(1.5)
            self._send_cmd_sync(sock, "SpeedFactor(100)")
            self._send_cmd_sync(sock, "SpeedJ(60)")
            self._send_cmd_sync(sock, "AccJ(60)")

        # Move both robots to home poses
        print(f"[+] Moving Left Robot to Target Pose: {self.left_home_pose}...")
        l_str = ",".join(f"{x:.4f}" for x in self.left_home_pose)
        self._send_cmd_sync(self.l_sock, f"JointMovJ({l_str})")

        print(f"[+] Moving Right Robot to Target Pose: {self.right_home_pose}...")
        r_str = ",".join(f"{x:.4f}" for x in self.right_home_pose)
        self._send_cmd_sync(self.r_sock, f"JointMovJ({r_str})")

        time.sleep(4.0)
        print("[+] Both Robot arms locked at home poses.")

    # ============================================================
    # STEP 2: RELEASE TORQUE FIRST -> RE-ENGAGE -> OPEN FULLY
    # ============================================================
    def step2_open_both_grippers_to_max(self):
        print("\n==================================================")
        print(" [STEP 2] TORQUE RELEASE -> CLEAN RE-ENGAGE -> MAX OPEN ")
        print("==================================================")

        # 1. Cleanly cut torque to reset servo stall status and unlock registers
        print("[+] Releasing holding torque on Left & Right grippers...")
        self.configure_microstep_gripper(side="both", enable=False)
        time.sleep(0.4)

        # 2. Re-enable PID holding torque with full register synchronization
        print("[+] Re-engaging calibrated 2% PID torque gains...")
        self.configure_microstep_gripper(side="both", enable=True)
        time.sleep(0.1)

        # 3. Drive Left Gripper to MAX OPEN (2000 ticks)
        print(f"[+] Homing Left Gripper to MAX OPEN ({self.L_GRIPPER_OPEN} ticks)...")
        self.move_percentage_microstep("left", 0.0, blocking=True)

        # 4. Drive Right Gripper to MAX OPEN (3865 ticks)
        print(f"[+] Homing Right Gripper to MAX OPEN ({self.R_GRIPPER_OPEN} ticks)...")
        self.move_percentage_microstep("right", 0.0, blocking=True)

        l_pos = self.read_gripper_pos("left")
        r_pos = self.read_gripper_pos("right")
        print(f"[+] Verified Final Positions -> Left: {l_pos} | Right: {r_pos} ticks.")
        print("[+] Both Grippers locked at MAX OPEN.")

    # ============================================================
    # STEP 3: DUAL PROXIMITY-GATED ALIGNMENT TIMER
    # ============================================================
    def step3_reposition_joysticks_with_prox_timer(self, target_seconds=10):
        print("\n==================================================")
        print(f" [STEP 3] BIMANUAL ALIGNMENT ({target_seconds}s HOLD REQUIRED) ")
        print("==================================================")
        print(">>> Grasp BOTH handles (triggers relaxed): Countdown runs when both are held.\n")

        accumulated_time = 0.0
        last_loop_time = time.time()
        last_blink_time = time.time()
        blink_state = False

        while accumulated_time < target_seconds:
            loop_now = time.time()
            dt = loop_now - last_loop_time
            last_loop_time = loop_now

            _, _, l_prox = self.get_official_keys(self.l_master_ph)
            _, _, r_prox = self.get_official_keys(self.r_master_ph)

            if l_prox and r_prox:
                accumulated_time += dt
                remaining = max(0.0, target_seconds - accumulated_time)
                timer_msg = f"\033[92m[COUNTING DOWN]\033[0m {remaining:04.1f}s left"

                if loop_now - last_blink_time > 0.3:
                    blink_state = not blink_state
                    self.set_leds("both", enable=blink_state)
                    last_blink_time = loop_now
            else:
                remaining = max(0.0, target_seconds - accumulated_time)
                status_parts = []
                if not l_prox:
                    status_parts.append("HOLD LEFT")
                if not r_prox:
                    status_parts.append("HOLD RIGHT")
                timer_msg = f"\033[93m[PAUSED: {' & '.join(status_parts)}]\033[0m {remaining:04.1f}s"
                self.set_leds("both", enable=False)

            sys.stdout.write(f"\r\033[KTIMER: {timer_msg:<48}")
            sys.stdout.flush()
            time.sleep(0.04)

        # Lock reference encoder baselines
        self.left_locked_ticks = self.read_encoders_safe(self.l_master_pkh, self.l_master_ph, self.left_joint_ids)
        self.right_locked_ticks = self.read_encoders_safe(self.r_master_pkh, self.r_master_ph, self.right_joint_ids)

        self.left_baseline_robot = np.array(self.left_home_pose, dtype=np.float64)
        self.right_baseline_robot = np.array(self.right_home_pose, dtype=np.float64)
        self.left_smoothed_joints = np.copy(self.left_baseline_robot)
        self.right_smoothed_joints = np.copy(self.right_baseline_robot)

        time.sleep(0.1)
        self.left_trigger_idle_ticks = self.read_trigger_safe(
            self.l_master_pkh, self.l_master_ph, self.left_trigger_id, 2048
        )
        self.right_trigger_idle_ticks = self.read_trigger_safe(
            self.r_master_pkh, self.r_master_ph, self.right_trigger_id, 1964
        )

        self.setup_master_haptic_modes()

        self.set_leds("both", enable=True)
        print(f"\n\n\033[92m[+] CALIBRATION COMPLETE! TELEOP ENGAGED WITH BALANCED HAPTICS!\033[0m")
        print(f"    Left Trigger Idle: {self.left_trigger_idle_ticks} | Right Trigger Idle: {self.right_trigger_idle_ticks}\n")

    # ============================================================
    # STEP 4: LIVE BIMANUAL 50 Hz TELEOP LOOP
    # ============================================================
    def step4_live_teleop(self):
        print("==================================================")
        print(">>> LIVE ZERO-LATENCY BIMANUAL TELEOP (SMOOTH & FAST 50 Hz)")
        print(f"    - Alpha Smoothing: {self.ARM_ALPHA} (Dynamic Trajectory)")
        print(f"    - Velocity Limit : {self.MAX_VELOCITY_DEG}° per step")
        print(f"    - Right Gripper  : ID 21 (Torque Limit: {self.R_TORQUE_LIMIT})")
        print("    - Green Button   : Toggle BRAKE on THAT arm")
        print("    - Yellow Button  : Toggle REPOSITION on THAT arm")
        print("==================================================\n")

        DT = 0.02
        l_prev_delta = np.zeros(6, dtype=np.float64)
        r_prev_delta = np.zeros(6, dtype=np.float64)
        l_was_holding = True
        r_was_holding = True
        loop_counter = 0

        try:
            while True:
                t0 = time.time()
                now = time.time()
                loop_counter += 1

                l_yel, l_grn, l_prox = self.get_official_keys(self.l_master_ph)
                r_yel, r_grn, r_prox = self.get_official_keys(self.r_master_ph)

                # ========================================================
                # 1. LEFT ARM BUTTON & MACRO LOGIC
                # ========================================================
                if l_grn and (now - self.left_last_btn_time > 0.4):
                    self.left_last_btn_time = now
                    if self.left_state == "BRAKED":
                        self.set_master_arm_torque("left", enable=False)
                        self.set_leds("left", enable=True)
                        time.sleep(0.02)
                        self.left_locked_ticks = self.read_encoders_safe(
                            self.l_master_pkh, self.l_master_ph, self.left_joint_ids, self.left_locked_ticks
                        )
                        self.left_baseline_robot = np.copy(self.left_smoothed_joints)
                        l_prev_delta = np.zeros(6, dtype=np.float64)
                        self.left_state = "RUNNING"
                    else:
                        self.left_state = "BRAKED"
                        self.set_master_arm_torque("left", enable=True)

                elif l_yel and (now - self.left_last_btn_time > 0.4):
                    self.left_last_btn_time = now
                    if self.left_state == "REPOSITION":
                        self.configure_microstep_gripper("left", enable=True)
                        self.setup_master_haptic_modes()
                        self.set_master_arm_torque("left", enable=False)
                        self.set_leds("left", enable=True)
                        time.sleep(0.02)
                        self.left_locked_ticks = self.read_encoders_safe(
                            self.l_master_pkh, self.l_master_ph, self.left_joint_ids, self.left_locked_ticks
                        )
                        self.left_baseline_robot = np.copy(self.left_smoothed_joints)
                        l_prev_delta = np.zeros(6, dtype=np.float64)
                        self.left_last_sent_gripper_reg = None
                        self.left_state = "RUNNING"
                    else:
                        self.left_state = "REPOSITION"
                        self.configure_microstep_gripper("left", enable=False)
                        self.set_master_arm_torque("left", enable=False)
                        try:
                            self.l_master_pkh.write1ByteTxRx(
                                self.l_master_ph, int(self.left_trigger_id), int(self.ADDR_TORQUE_ENABLE), 0
                            )
                        except Exception:
                            pass
                        self.set_leds("left", enable=False)

                # ========================================================
                # 2. RIGHT ARM BUTTON & MACRO LOGIC
                # ========================================================
                if r_grn and (now - self.right_last_btn_time > 0.4):
                    self.right_last_btn_time = now
                    if self.right_state == "BRAKED":
                        self.set_master_arm_torque("right", enable=False)
                        self.set_leds("right", enable=True)
                        time.sleep(0.02)
                        self.right_locked_ticks = self.read_encoders_safe(
                            self.r_master_pkh, self.r_master_ph, self.right_joint_ids, self.right_locked_ticks
                        )
                        self.right_baseline_robot = np.copy(self.right_smoothed_joints)
                        r_prev_delta = np.zeros(6, dtype=np.float64)
                        self.right_state = "RUNNING"
                    else:
                        self.right_state = "BRAKED"
                        self.set_master_arm_torque("right", enable=True)

                elif r_yel and (now - self.right_last_btn_time > 0.4):
                    self.right_last_btn_time = now
                    if self.right_state == "REPOSITION":
                        self.configure_microstep_gripper("right", enable=True)
                        self.setup_master_haptic_modes()
                        self.set_master_arm_torque("right", enable=False)
                        self.set_leds("right", enable=True)
                        time.sleep(0.02)
                        self.right_locked_ticks = self.read_encoders_safe(
                            self.r_master_pkh, self.r_master_ph, self.right_joint_ids, self.right_locked_ticks
                        )
                        self.right_baseline_robot = np.copy(self.right_smoothed_joints)
                        r_prev_delta = np.zeros(6, dtype=np.float64)
                        self.right_last_sent_gripper_reg = None
                        self.right_state = "RUNNING"
                    else:
                        self.right_state = "REPOSITION"
                        self.configure_microstep_gripper("right", enable=False)
                        self.set_master_arm_torque("right", enable=False)
                        try:
                            self.r_master_pkh.write1ByteTxRx(
                                self.r_master_ph, int(self.right_trigger_id), int(self.ADDR_TORQUE_ENABLE), 0
                            )
                        except Exception:
                            pass
                        self.set_leds("right", enable=False)

                # Proximity Re-sync
                if l_prox and not l_was_holding:
                    self.left_locked_ticks = self.read_encoders_safe(
                        self.l_master_pkh, self.l_master_ph, self.left_joint_ids, self.left_locked_ticks
                    )
                    self.left_baseline_robot = np.copy(self.left_smoothed_joints)
                    l_prev_delta = np.zeros(6, dtype=np.float64)
                l_was_holding = l_prox

                if r_prox and not r_was_holding:
                    self.right_locked_ticks = self.read_encoders_safe(
                        self.r_master_pkh, self.r_master_ph, self.right_joint_ids, self.right_locked_ticks
                    )
                    self.right_baseline_robot = np.copy(self.right_smoothed_joints)
                    r_prev_delta = np.zeros(6, dtype=np.float64)
                r_was_holding = r_prox

                # ========================================================
                # 3. TRIGGER & GRIPPER PROCESSING (ZERO DELAY)
                # ========================================================
                l_trig_now = self.read_trigger_safe(
                    self.l_master_pkh, self.l_master_ph, self.left_trigger_id, self.left_trigger_idle_ticks
                )
                l_disp = abs(l_trig_now - self.left_trigger_idle_ticks)
                l_pull = max(0, l_disp - self.TRIGGER_DEADBAND)
                l_pct = np.clip(l_pull / self.L_TRIGGER_TRAVEL, 0.0, 1.0)
                l_target_pos = self.L_GRIPPER_OPEN
                if self.left_state == "RUNNING":
                    l_target_pos = self.move_percentage_microstep("left", l_pct, blocking=False)

                r_trig_now = self.read_trigger_safe(
                    self.r_master_pkh, self.r_master_ph, self.right_trigger_id, self.right_trigger_idle_ticks
                )
                r_disp = abs(r_trig_now - self.right_trigger_idle_ticks)
                r_pull = max(0, r_disp - self.TRIGGER_DEADBAND)
                r_pct = np.clip(r_pull / self.R_TRIGGER_TRAVEL, 0.0, 1.0)
                r_target_pos = self.R_GRIPPER_OPEN
                if self.right_state == "RUNNING":
                    r_target_pos = self.move_percentage_microstep("right", r_pct, blocking=False)

                # ========================================================
                # 4. BILATERAL HAPTICS (INTERLEAVED BUS ACCESS)
                # ========================================================
                l_haptic_mA = 0
                r_haptic_mA = 0
                if loop_counter % 2 == 0:
                    l_load = self.read_gripper_load("left")
                    if self.left_state == "RUNNING":
                        l_haptic_mA = self.apply_smooth_trigger_haptics("left", l_load)
                else:
                    r_load = self.read_gripper_load("right")
                    if self.right_state == "RUNNING":
                        r_haptic_mA = self.apply_smooth_trigger_haptics("right", r_load)

                # ========================================================
                # 5. ROBOT ARMS MOTION STREAMING (ZERO-LATENCY 50 Hz)
                # ========================================================
                if (self.left_state == "RUNNING") and l_prox:
                    l_ticks = self.read_encoders_safe(
                        self.l_master_pkh, self.l_master_ph, self.left_joint_ids, self.left_locked_ticks
                    )
                    l_raw_delta = (l_ticks - self.left_locked_ticks) * (360.0 / 4096.0) * self.left_joint_signs

                    l_diff = np.abs(l_raw_delta - l_prev_delta)
                    l_act_delta = np.where(l_diff < self.DEADBAND_DEG, l_prev_delta, l_raw_delta)
                    l_prev_delta = l_act_delta

                    l_raw_target = self.left_baseline_robot + l_act_delta
                    l_step = np.clip(
                        l_raw_target - self.left_smoothed_joints, -self.MAX_VELOCITY_DEG, self.MAX_VELOCITY_DEG
                    )
                    self.left_smoothed_joints = (
                        self.ARM_ALPHA * (self.left_smoothed_joints + l_step)
                        + (1.0 - self.ARM_ALPHA) * self.left_smoothed_joints
                    )
                    self._send_servoj_fast(self.l_sock, self.left_smoothed_joints)

                if (self.right_state == "RUNNING") and r_prox:
                    r_ticks = self.read_encoders_safe(
                        self.r_master_pkh, self.r_master_ph, self.right_joint_ids, self.right_locked_ticks
                    )
                    r_raw_delta = (r_ticks - self.right_locked_ticks) * (360.0 / 4096.0) * self.right_joint_signs

                    r_diff = np.abs(r_raw_delta - r_prev_delta)
                    r_act_delta = np.where(r_diff < self.DEADBAND_DEG, r_prev_delta, r_raw_delta)
                    r_prev_delta = r_act_delta

                    r_raw_target = self.right_baseline_robot + r_act_delta
                    r_step = np.clip(
                        r_raw_target - self.right_smoothed_joints, -self.MAX_VELOCITY_DEG, self.MAX_VELOCITY_DEG
                    )
                    self.right_smoothed_joints = (
                        self.ARM_ALPHA * (self.right_smoothed_joints + r_step)
                        + (1.0 - self.ARM_ALPHA) * self.right_smoothed_joints
                    )
                    self._send_servoj_fast(self.r_sock, self.right_smoothed_joints)

                # ========================================================
                # 6. LIVE BIMANUAL STATUS BAR
                # ========================================================
                def get_status_tag(state, prox):
                    if state == "BRAKED":
                        return "\033[91m[BRAKED]\033[0m"
                    elif state == "REPOSITION":
                        return "\033[93m[REPOS]\033[0m"
                    elif prox:
                        return "\033[92m[ACTIVE]\033[0m"
                    else:
                        return "\033[90m[STANDBY]\033[0m"

                l_tag = get_status_tag(self.left_state, l_prox)
                r_tag = get_status_tag(self.right_state, r_prox)

                l_grip_pct = int(round(self.left_smoothed_squeeze_pct * 100))
                r_grip_pct = int(round(self.right_smoothed_squeeze_pct * 100))
                l_hap_str = f"\033[96m{l_haptic_mA:3d}mA\033[0m" if l_haptic_mA > 0 else "0mA"
                r_hap_str = f"\033[96m{r_haptic_mA:3d}mA\033[0m" if r_haptic_mA > 0 else "0mA"

                sys.stdout.write(
                    f"\r\033[KL: {l_tag:<18} | SQ:{l_grip_pct:3d}% (Pos:{l_target_pos:4d}) | HAP:{l_hap_str:<12} | "
                    f"R: {r_tag:<18} | SQ:{r_grip_pct:3d}% (Pos:{r_target_pos:4d}) | HAP:{r_hap_str:<12}"
                )
                sys.stdout.flush()

                elapsed = time.time() - t0
                if elapsed < DT:
                    time.sleep(DT - elapsed)

        except KeyboardInterrupt:
            print("\n\n[-] Bimanual teleoperation ended by user (Ctrl+C).")
        finally:
            self.shutdown()

    def shutdown(self):
        print("\n[+] Releasing all torques and shutting down cleanly...")
        try:
            self.set_leds("both", enable=False)
            self.set_master_arm_torque("both", enable=False)
            for pkh, ph, trig_id in [
                (self.l_master_pkh, self.l_master_ph, self.left_trigger_id),
                (self.r_master_pkh, self.r_master_ph, self.right_trigger_id),
            ]:
                try:
                    pkh.write1ByteTxRx(ph, int(trig_id), int(self.ADDR_TORQUE_ENABLE), 0)
                except Exception:
                    pass

            self.release_gripper_torque_clean(side="both")

            if self.l_master_ph:
                self.l_master_ph.closePort()
            if self.r_master_ph:
                self.r_master_ph.closePort()
            if self.l_grip_ph:
                self.l_grip_ph.closePort()
            if self.r_grip_ph:
                self.r_grip_ph.closePort()

            for name, sock in [("Left", self.l_sock), ("Right", self.r_sock)]:
                if sock:
                    self._send_cmd_sync(sock, "DisableRobot()")
                    sock.close()
        except Exception:
            pass
        print("[+] Dual-arm system safely offline.")


if __name__ == "__main__":
    app = DualArmBimanualPrecisionTeleop(
        left_robot_ip="192.168.5.1",
        right_robot_ip="192.168.5.2",
        left_master_port="COM5",
        right_master_port="COM3",
        left_gripper_port="COM6",
        right_gripper_port="COM7",
        master_baudrate=2000000,
        gripper_baudrate=1000000,
    )
    try:
        app.step1_travel_both_robots_to_coordinates()
        app.step2_open_both_grippers_to_max()
        app.step3_reposition_joysticks_with_prox_timer(target_seconds=10)
        app.step4_live_teleop()
    except Exception as err:
        print(f"\n[-] Execution Error: {err}")
        app.shutdown()