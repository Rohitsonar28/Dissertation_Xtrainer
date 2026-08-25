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


class LeftArmAndPrecisionGripperTeleop:
    def __init__(
        self,
        robot_ip="192.168.5.1",
        master_port="COM5",
        gripper_port="COM6",
        master_baudrate=2000000,
        gripper_baudrate=1000000,
    ):
        self.robot_ip = robot_ip
        self.master_port = master_port
        self.gripper_port = gripper_port
        self.master_baudrate = int(master_baudrate)
        self.gripper_baudrate = int(gripper_baudrate)

        # ------------------------------------------------------------
        # EXACT LEFT ARM DATA FROM SOURCE FILE
        # ------------------------------------------------------------
        # Initial Target Robot Pose (Degrees)
        self.home_pose = [-89.7758, 1.3375, -86.5728, -2.3068, 87.0675, -0.4531]

        # 6-DOF Active Joint IDs & Master IDs
        self.joint_ids = [1, 2, 4, 5, 6, 7]
        self.all_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        self.arm_ids = [1, 2, 3, 4, 5, 6, 7]
        self.trigger_id = 8

        # Kinematic Direction Signs (J1 Inverted on Left for Intuitive Push/Pull)
        self.joint_signs = np.array([1, 1, -1, -1, -1, 1], dtype=np.float64)

        # Dynamic Filter Parameters
        self.ARM_ALPHA = 0.38
        self.DEADBAND_DEG = 0.10
        self.MAX_VELOCITY_DEG = 8.0

        # Dynamixel Protocol 2.0 Registers (Master Handle)
        self.ADDR_OPERATING_MODE = 11
        self.ADDR_TORQUE_ENABLE = 64
        self.ADDR_LED = 65
        self.ADDR_GOAL_CURRENT = 102
        self.ADDR_GOAL_POSITION = 116
        self.ADDR_PRESENT_POSITION = 132

        # ------------------------------------------------------------
        # CALIBRATED LEFT GRIPPER LIMITS (Feetech STS ID 22 on COM6)
        # ------------------------------------------------------------
        self.GRIPPER_ID = 22
        self.BROADCAST_ID = 254
        self.GRIPPER_OPEN = 2000       # 0% Squeeze (Open)
        self.GRIPPER_CLOSED = 980      # 100% Squeeze (Closed)
        self.TOTAL_STROKE = self.GRIPPER_OPEN - self.GRIPPER_CLOSED  # 1020 ticks

        # 2% Micro-Precision Trigger Mapping
        self.PRECISION_BUCKETS = 50.0  # 2% resolution quantization
        self.GRIP_ALPHA = 0.50
        self.smoothed_squeeze_pct = 0.0
        self.TRIGGER_TRAVEL_TICKS = 240.0
        self.TRIGGER_DEADBAND = 6.0

        # Continuous Smooth Haptics (Mode 5 Current-Position Feedback)
        self.HAPTIC_INTENSITY = 0.65
        self.LOAD_DEADBAND = 40
        self.MAX_FEEDBACK_CURRENT = 260
        self.HAPTIC_SMOOTH_ALPHA = 0.25
        self.filtered_haptic_current = 0.0
        self.last_sent_current = None

        # Hardware Handlers
        self.master_ph = None
        self.master_pkh = None
        self.grip_ph = None
        self.grip_pkh = None
        self.sock = None

        # State Tracking
        self.baseline_robot = np.array(self.home_pose, dtype=np.float64)
        self.locked_ticks = None
        self.smoothed_joints = np.array(self.home_pose, dtype=np.float64)

        self.trigger_idle_ticks = 2048
        self.last_sent_gripper_reg = None
        self.last_known_gripper_pos = self.GRIPPER_OPEN
        self.last_known_gripper_load = 0

        self.state = "RUNNING"  # RUNNING | REPOSITION | BRAKED
        self.last_btn_time = 0.0

    # ------------------------------------------------------------
    # LOW-LEVEL ROBOT COMMS
    # ------------------------------------------------------------
    def _send_cmd_sync(self, cmd_str):
        try:
            msg = (cmd_str + "\r\n").encode("utf-8")
            self.sock.sendall(msg)
            time.sleep(0.05)
            return self.sock.recv(1024).decode("utf-8").strip()
        except Exception as e:
            return f"Error: {e}"

    def _send_servoj_fast(self, joint_angles):
        j_str = ",".join(f"{float(x):.4f}" for x in joint_angles)
        msg = f"ServoJ({j_str},t=0.02)\r\n".encode("utf-8")
        try:
            self.sock.sendall(msg)
            self.sock.setblocking(False)
            try:
                self.sock.recv(1024)
            except (BlockingIOError, socket.error):
                pass
            self.sock.setblocking(True)
        except Exception:
            pass

    # ------------------------------------------------------------
    # MASTER HARDWARE METHODS & SMOOTH HAPTICS
    # ------------------------------------------------------------
    def set_master_arm_torque(self, enable=False):
        val = 1 if enable else 0
        try:
            for dxl_id in self.arm_ids:
                self.master_pkh.write1ByteTxRx(
                    self.master_ph, int(dxl_id), int(self.ADDR_TORQUE_ENABLE), val
                )
        except Exception:
            pass

    def setup_master_haptic_mode(self):
        """Puts Left trigger into Mode 5 (Current-based Position Mode) for smooth spring resistance."""
        try:
            self.master_pkh.write1ByteTxRx(self.master_ph, int(self.trigger_id), int(self.ADDR_TORQUE_ENABLE), 0)
            time.sleep(0.02)
            self.master_pkh.write1ByteTxRx(self.master_ph, int(self.trigger_id), int(self.ADDR_OPERATING_MODE), 5)
            self.master_pkh.write1ByteTxRx(self.master_ph, int(self.trigger_id), int(self.ADDR_TORQUE_ENABLE), 1)
            time.sleep(0.02)
        except Exception:
            pass

    def set_leds(self, enable=True):
        val = 1 if enable else 0
        try:
            for dxl_id in self.all_ids:
                self.master_pkh.write1ByteTxRx(
                    self.master_ph, int(dxl_id), int(self.ADDR_LED), val
                )
        except Exception:
            pass

    def read_encoders_safe(self, fallback_ticks=None):
        raw_ticks = []
        for idx, dxl_id in enumerate(self.joint_ids):
            default_val = float(fallback_ticks[idx]) if fallback_ticks is not None else 2048.0
            try:
                pos, res, _ = self.master_pkh.read4ByteTxRx(
                    self.master_ph, int(dxl_id), int(self.ADDR_PRESENT_POSITION)
                )
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

    def read_trigger_safe(self):
        try:
            pos, res, _ = self.master_pkh.read4ByteTxRx(
                self.master_ph, int(self.trigger_id), int(self.ADDR_PRESENT_POSITION)
            )
            if res != 0:
                return self.trigger_idle_ticks
            pos = int(pos) & 0xFFFFFFFF
            if pos > 0x7FFFFFFF:
                pos -= 0x100000000
            return pos
        except Exception:
            return self.trigger_idle_ticks

    def apply_smooth_trigger_haptics(self, slave_load):
        load_mag = slave_load & 0x03FF
        if load_mag > self.LOAD_DEADBAND:
            excess_load = load_mag - self.LOAD_DEADBAND
            raw_target_current = float(excess_load * self.HAPTIC_INTENSITY)
            raw_target_current = min(float(self.MAX_FEEDBACK_CURRENT), raw_target_current)
        else:
            raw_target_current = 0.0

        self.filtered_haptic_current = (
            self.HAPTIC_SMOOTH_ALPHA * raw_target_current + (1.0 - self.HAPTIC_SMOOTH_ALPHA) * self.filtered_haptic_current
        )
        target_int = int(round(self.filtered_haptic_current))

        if self.last_sent_current is None or abs(target_int - self.last_sent_current) >= 2:
            try:
                self.master_pkh.write4ByteTxRx(
                    self.master_ph, int(self.trigger_id), int(self.ADDR_GOAL_POSITION), int(self.trigger_idle_ticks)
                )
                self.master_pkh.write2ByteTxRx(
                    self.master_ph, int(self.trigger_id), int(self.ADDR_GOAL_CURRENT), target_int
                )
                self.last_sent_current = target_int
            except Exception:
                pass

        return target_int

    def get_official_keys(self):
        try:
            self.master_ph.clearPort()
            self.master_ph.writePort([0xAA, 0x55, 0xAA])

            rx = []
            t0 = time.time()
            while len(rx) < 4 and (time.time() - t0) < 0.02:
                chunk = self.master_ph.readPort(int(4 - len(rx)))
                if chunk:
                    rx.extend(chunk)

            if len(rx) >= 4 and (int(rx[0]) + int(rx[1]) + int(rx[2]) + int(rx[3])) == (255 * 2):
                key1 = (int(rx[0]) >> 4) & 0x0F  # Yellow Button (0 = Pressed)
                key2 = int(rx[0]) & 0x0F         # Green Button (0 = Pressed)
                prox = int(rx[2])                # Proximity Sensor (0 = Hand Covered)
                return (key1 == 0), (key2 == 0), (prox == 0)
        except Exception:
            pass
        return False, False, False

    # ------------------------------------------------------------
    # 2% MICRO-STEPPING GRIPPER CONTROL
    # ------------------------------------------------------------
    def read_gripper_pos(self):
        try:
            pos, res, _ = self.grip_pkh.read2ByteTxRx(self.grip_ph, self.GRIPPER_ID, 56)
            if res == 0 and pos is not None:
                self.last_known_gripper_pos = pos
                return pos
        except Exception:
            pass
        return self.last_known_gripper_pos

    def read_gripper_load(self):
        try:
            load, res, _ = self.grip_pkh.read2ByteTxRx(self.grip_ph, self.GRIPPER_ID, 60)
            if res == 0 and load is not None:
                self.last_known_gripper_load = load & 0x03FF
                return self.last_known_gripper_load
        except Exception:
            pass
        return self.last_known_gripper_load

    def configure_microstep_gripper(self, enable=True):
        if not self.grip_ph:
            return
        try:
            if enable:
                self.grip_pkh.write1ByteTxRx(self.grip_ph, self.GRIPPER_ID, 55, 0)
                time.sleep(0.02)
                self.grip_pkh.write1ByteTxRx(self.grip_ph, self.GRIPPER_ID, 33, 0)      # Position Mode
                self.grip_pkh.write2ByteTxRx(self.grip_ph, self.GRIPPER_ID, 9, 0)
                self.grip_pkh.write2ByteTxRx(self.grip_ph, self.GRIPPER_ID, 11, 4095)

                # 2% Micro-Step PID Registers
                self.grip_pkh.write1ByteTxRx(self.grip_ph, self.GRIPPER_ID, 21, 36)
                self.grip_pkh.write1ByteTxRx(self.grip_ph, self.GRIPPER_ID, 22, 14)
                self.grip_pkh.write1ByteTxRx(self.grip_ph, self.GRIPPER_ID, 23, 2)
                self.grip_pkh.write1ByteTxRx(self.grip_ph, self.GRIPPER_ID, 24, 1)

                self.grip_pkh.write2ByteTxRx(self.grip_ph, self.GRIPPER_ID, 48, 850)
                self.grip_pkh.write2ByteTxRx(self.grip_ph, self.GRIPPER_ID, 44, 2600)
                self.grip_pkh.write1ByteTxRx(self.grip_ph, self.GRIPPER_ID, 41, 70)
                self.grip_pkh.write1ByteTxRx(self.grip_ph, self.GRIPPER_ID, 40, 1)
                time.sleep(0.05)
            else:
                for t_id in [self.GRIPPER_ID, self.BROADCAST_ID]:
                    self.grip_pkh.write1ByteTxRx(self.grip_ph, t_id, 40, 0)
                    self.grip_pkh.write2ByteTxRx(self.grip_ph, t_id, 48, 0)
        except Exception:
            pass

    def move_percentage_microstep(self, raw_pct, blocking=False):
        raw_pct = float(np.clip(raw_pct, 0.0, 1.0))
        quantized_pct = np.round(raw_pct * self.PRECISION_BUCKETS) / self.PRECISION_BUCKETS
        self.smoothed_squeeze_pct = self.GRIP_ALPHA * quantized_pct + (1.0 - self.GRIP_ALPHA) * self.smoothed_squeeze_pct

        target = int(round(self.GRIPPER_OPEN - self.smoothed_squeeze_pct * self.TOTAL_STROKE))

        if not blocking:
            if self.last_sent_gripper_reg is None or abs(target - self.last_sent_gripper_reg) >= 1:
                try:
                    self.grip_pkh.write2ByteTxRx(self.grip_ph, self.GRIPPER_ID, 42, target)
                    self.last_sent_gripper_reg = target
                except Exception:
                    pass
            return target

        try:
            self.grip_pkh.write2ByteTxRx(self.grip_ph, self.GRIPPER_ID, 42, target)
            self.last_sent_gripper_reg = target
        except Exception:
            pass

        t0 = time.time()
        while time.time() - t0 < 2.5:
            curr = self.read_gripper_pos()
            if curr is not None:
                sys.stdout.write(f"\r\033[K    Moving to: {target:4d} | Current: {curr:4d} ticks")
                sys.stdout.flush()
                if abs(curr - target) <= 2:
                    break
            time.sleep(0.02)
        print()
        return target

    # ============================================================
    # STEP 1: INITIALIZE HARDWARE & MOVE ARM FIRST
    # ============================================================
    def step1_move_arm_to_position(self):
        print("\n==================================================")
        print(" [STEP 1] INITIALIZING HARDWARE & MOVING LEFT ARM ")
        print("==================================================")

        print(f"[+] Connecting Left Master Joystick on {self.master_port}...")
        self.master_ph = PortHandler(self.master_port)
        self.master_pkh = PacketHandler(2.0)
        if not self.master_ph.openPort() or not self.master_ph.setBaudRate(self.master_baudrate):
            raise RuntimeError(f"Could not open {self.master_port}")

        self.set_master_arm_torque(enable=False)
        self.set_leds(enable=False)

        print(f"[+] Connecting Left Gripper on {self.gripper_port}...")
        self.grip_ph = PortHandler(self.gripper_port)
        self.grip_pkh = PacketHandler(1.0)
        if not self.grip_ph.openPort() or not self.grip_ph.setBaudRate(self.gripper_baudrate):
            raise RuntimeError(f"Could not open {self.gripper_port}")

        print(f"\n[+] Connecting Left Nova 2 at {self.robot_ip}:29999...")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(5.0)
        self.sock.connect((self.robot_ip, 29999))

        self._send_cmd_sync("ClearError()")
        time.sleep(0.5)
        self._send_cmd_sync("PowerOn()")
        time.sleep(3.0)
        self._send_cmd_sync("EnableRobot()")
        time.sleep(1.5)
        mode_str = self._send_cmd_sync("RobotMode()")
        if "{5}" not in mode_str:
            self._send_cmd_sync("EnableRobot(0)")
            time.sleep(1.5)

        self._send_cmd_sync("SpeedFactor(100)")
        self._send_cmd_sync("SpeedJ(50)")
        self._send_cmd_sync("AccJ(50)")

        print(f"[+] Moving Left Arm to Initial Target Pose: {self.home_pose}...")
        r_str = ",".join(f"{x:.4f}" for x in self.home_pose)
        self._send_cmd_sync(f"JointMovJ({r_str})")
        time.sleep(4.0)
        print("[+] Left robot arm arrived and locked at initial pose.")

    # ============================================================
    # STEP 2: APPLY 2% COMPLIANCE & OPEN GRIPPER TO 2000 TICKS
    # ============================================================
    def step2_open_gripper_to_max(self):
        print("\n==================================================")
        print(" [STEP 2] 2% GAINS & OPENING GRIPPER TO 2000      ")
        print("==================================================")

        self.configure_microstep_gripper(enable=False)
        time.sleep(0.2)
        self.configure_microstep_gripper(enable=True)
        time.sleep(0.05)

        raw_curr = self.read_gripper_pos()
        print(f"[+] Current Gripper Position: {raw_curr} ticks")
        print(f"[+] Homing to MAX OPEN ({self.GRIPPER_OPEN} ticks)...")

        self.move_percentage_microstep(0.0, blocking=True)
        print(f"[+] Gripper locked at MAX OPEN ({self.read_gripper_pos()} ticks).")

    # ============================================================
    # STEP 3: PROXIMITY ALIGNMENT TIMER
    # ============================================================
    def step3_reposition_joystick_with_prox_timer(self, target_seconds=10):
        print("\n==================================================")
        print(f" [STEP 3] JOYSTICK ALIGNMENT ({target_seconds}s HOLD REQUIRED)  ")
        print("==================================================")
        print(">>> Grasp the Left handle (keep trigger relaxed): Countdown runs when held.\n")

        accumulated_time = 0.0
        last_loop_time = time.time()
        last_blink_time = time.time()
        blink_state = False

        while accumulated_time < target_seconds:
            loop_now = time.time()
            dt = loop_now - last_loop_time
            last_loop_time = loop_now

            _, _, prox = self.get_official_keys()

            if prox:
                accumulated_time += dt
                remaining = max(0.0, target_seconds - accumulated_time)
                timer_msg = f"\033[92m[COUNTING DOWN]\033[0m {remaining:04.1f}s left"

                if loop_now - last_blink_time > 0.3:
                    blink_state = not blink_state
                    self.set_leds(enable=blink_state)
                    last_blink_time = loop_now
            else:
                remaining = max(0.0, target_seconds - accumulated_time)
                timer_msg = f"\033[93m[PAUSED: HOLD HANDLE]\033[0m {remaining:04.1f}s"
                self.set_leds(enable=False)

            sys.stdout.write(f"\r\033[KTIMER: {timer_msg:<40}")
            sys.stdout.flush()
            time.sleep(0.04)

        self.locked_ticks = self.read_encoders_safe()
        self.baseline_robot = np.array(self.home_pose, dtype=np.float64)
        self.smoothed_joints = np.copy(self.baseline_robot)

        time.sleep(0.1)
        self.trigger_idle_ticks = self.read_trigger_safe()
        self.setup_master_haptic_mode()

        self.set_leds(enable=True)
        print(f"\n\n\033[92m[+] CALIBRATION COMPLETE! TRIGGER IDLE BASELINE: {self.trigger_idle_ticks}\033[0m\n")

    # ============================================================
    # STEP 4: LIVE 50 Hz TELEOP
    # ============================================================
    def step4_live_teleop(self):
        print("==================================================")
        print(">>> LIVE LEFT ARM + 2% MICRO-PRECISION TELEOP (50 Hz)")
        print(f"    - Joint IDs       : {self.joint_ids}")
        print(f"    - Joint Signs     : {self.joint_signs.tolist()}")
        print(f"    - Trigger Idle    : OPEN  ({self.GRIPPER_OPEN} ticks)")
        print(f"    - Trigger Squeeze : CLOSE ({self.GRIPPER_CLOSED} ticks)")
        print("    - Yellow Button   : [REPOSITION CLUTCH] -> Free-Hand Adjust")
        print("    - Green Button    : Toggle Brake on Arm")
        print("==================================================\n")

        DT = 0.02
        prev_delta = np.zeros(6, dtype=np.float64)
        was_holding = True

        try:
            while True:
                t0 = time.time()
                now = time.time()

                yel, grn, prox = self.get_official_keys()

                # --- GREEN BUTTON: ARM BRAKE TOGGLE ---
                if grn and (now - self.last_btn_time > 0.4):
                    self.last_btn_time = now
                    if self.state == "BRAKED":
                        self.set_master_arm_torque(enable=False)
                        self.set_leds(enable=True)
                        time.sleep(0.02)
                        self.locked_ticks = self.read_encoders_safe(self.locked_ticks)
                        self.baseline_robot = np.copy(self.smoothed_joints)
                        prev_delta = np.zeros(6, dtype=np.float64)
                        self.state = "RUNNING"
                    else:
                        self.state = "BRAKED"
                        self.set_master_arm_torque(enable=True)

                # --- YELLOW BUTTON: REPOSITION CLUTCH ---
                elif yel and (now - self.last_btn_time > 0.4):
                    self.last_btn_time = now
                    if self.state == "REPOSITION":
                        print("\n[+] Resuming Teleoperation & Re-enabling Gripper...")
                        self.configure_microstep_gripper(enable=True)
                        self.setup_master_haptic_mode()
                        self.set_master_arm_torque(enable=False)
                        self.set_leds(enable=True)
                        time.sleep(0.02)
                        self.locked_ticks = self.read_encoders_safe(self.locked_ticks)
                        self.baseline_robot = np.copy(self.smoothed_joints)
                        prev_delta = np.zeros(6, dtype=np.float64)
                        self.last_sent_gripper_reg = None
                        self.state = "RUNNING"
                    else:
                        print("\n[!] REPOSITION MODE: Gripper & Arm Torque Released (Holding Last Pos)")
                        self.state = "REPOSITION"
                        self.configure_microstep_gripper(enable=False)
                        self.set_master_arm_torque(enable=False)
                        try:
                            self.master_pkh.write1ByteTxRx(self.master_ph, int(self.trigger_id), int(self.ADDR_TORQUE_ENABLE), 0)
                        except Exception:
                            pass
                        self.set_leds(enable=False)

                # --- PROXIMITY RE-SYNC ---
                if prox and not was_holding:
                    self.locked_ticks = self.read_encoders_safe(self.locked_ticks)
                    self.baseline_robot = np.copy(self.smoothed_joints)
                    prev_delta = np.zeros(6, dtype=np.float64)
                was_holding = prox

                # --- 1. PROPORTIONAL TRIGGER READING ---
                trig_now = self.read_trigger_safe()
                displacement = abs(trig_now - self.trigger_idle_ticks)
                active_pull = max(0, displacement - self.TRIGGER_DEADBAND)
                squeeze_pct = np.clip(active_pull / self.TRIGGER_TRAVEL_TICKS, 0.0, 1.0)

                target_pos = self.GRIPPER_OPEN
                if self.state == "RUNNING":
                    target_pos = self.move_percentage_microstep(squeeze_pct, blocking=False)

                # --- 2. BILATERAL HAPTICS ---
                slave_load = self.read_gripper_load()
                haptic_force_mA = 0
                if self.state == "RUNNING":
                    haptic_force_mA = self.apply_smooth_trigger_haptics(slave_load)

                # --- 3. ROBOT ARM MOTION STREAMING (50 Hz) ---
                if (self.state == "RUNNING") and prox:
                    ticks = self.read_encoders_safe(self.locked_ticks)
                    raw_delta = (ticks - self.locked_ticks) * (360.0 / 4096.0) * self.joint_signs

                    diff = np.abs(raw_delta - prev_delta)
                    act_delta = np.where(diff < self.DEADBAND_DEG, prev_delta, raw_delta)
                    prev_delta = act_delta

                    raw_target = self.baseline_robot + act_delta
                    step = np.clip(raw_target - self.smoothed_joints, -self.MAX_VELOCITY_DEG, self.MAX_VELOCITY_DEG)
                    self.smoothed_joints = self.ARM_ALPHA * (self.smoothed_joints + step) + (1.0 - self.ARM_ALPHA) * self.smoothed_joints
                    self._send_servoj_fast(self.smoothed_joints)

                # --- 4. LIVE STATUS BAR ---
                if self.state == "BRAKED":
                    tag = "\033[91m[ARM BRAKED]\033[0m"
                elif self.state == "REPOSITION":
                    tag = "\033[93m[REPOS: TORQUE OFF]\033[0m"
                else:
                    tag = "\033[92m[ACTIVE]\033[0m" if prox else "\033[90m[STANDBY]\033[0m"

                grip_int_pct = int(round(self.smoothed_squeeze_pct * 100))
                haptic_tag = f"\033[96m{haptic_force_mA:3d}mA\033[0m" if haptic_force_mA > 0 else "0mA"

                sys.stdout.write(
                    f"\r\033[KSTATUS: {tag:<22} | SQUEEZE: {grip_int_pct:3d}% (Pos: {target_pos:4d}) | "
                    f"LOAD: {slave_load & 0x03FF:4d} | HAPTIC: {haptic_tag:<14} | "
                    f"J1:{self.smoothed_joints[0]:+06.1f}° J2:{self.smoothed_joints[1]:+06.1f}°"
                )
                sys.stdout.flush()

                elapsed = time.time() - t0
                if elapsed < DT:
                    time.sleep(DT - elapsed)

        except KeyboardInterrupt:
            print("\n\n[-] Teleoperation stopped by user (Ctrl+C).")
        finally:
            self.shutdown()

    def shutdown(self):
        print("\n[+] Releasing all torques (Arm + Gripper + Haptics) and closing ports...")
        try:
            self.set_leds(enable=False)
            self.set_master_arm_torque(enable=False)
            try:
                self.master_pkh.write1ByteTxRx(self.master_ph, int(self.trigger_id), int(self.ADDR_TORQUE_ENABLE), 0)
            except Exception:
                pass
            self.configure_microstep_gripper(enable=False)
            if self.master_ph:
                self.master_ph.closePort()
            if self.grip_ph:
                self.grip_ph.closePort()
            if self.sock:
                self._send_cmd_sync("DisableRobot()")
                self.sock.close()
        except Exception:
            pass
        print("[+] System safely offline.")


if __name__ == "__main__":
    app = LeftArmAndPrecisionGripperTeleop(
        robot_ip="192.168.5.1",
        master_port="COM5",
        gripper_port="COM6",
        master_baudrate=2000000,
        gripper_baudrate=1000000,
    )
    try:
        app.step1_move_arm_to_position()
        app.step2_open_gripper_to_max()
        app.step3_reposition_joystick_with_prox_timer(target_seconds=10)
        app.step4_live_teleop()
    except Exception as err:
        print(f"\n[-] Execution Error: {err}")
        app.shutdown()