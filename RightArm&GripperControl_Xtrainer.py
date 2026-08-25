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


class RightArmAndPrecisionGripperTeleop:
    def __init__(
        self,
        robot_ip="192.168.5.2",
        master_port="COM3",
        gripper_port="COM7",
        master_baudrate=2000000,
        gripper_baudrate=1000000,
    ):
        self.robot_ip = robot_ip
        self.master_port = master_port
        self.gripper_port = gripper_port
        self.master_baudrate = int(master_baudrate)
        self.gripper_baudrate = int(gripper_baudrate)

        # Right Arm Initial Pose (Degrees)
        self.home_pose = [89.7758, 1.3375, 86.5728, 2.3068, -87.0675, 0.4531]

        # Master Joint IDs & Signs
        self.joint_ids = [11, 12, 14, 15, 16, 17]
        self.arm_ids = [11, 12, 13, 14, 15, 16, 17]
        self.all_ids = [11, 12, 13, 14, 15, 16, 17, 18]
        self.trigger_id = 18
        self.joint_signs = np.array([1, 1, -1, -1, -1, 1], dtype=np.float64)

        # Dynamixel Protocol 2.0 Registers (Master Handle)
        self.ADDR_TORQUE_ENABLE = 64
        self.ADDR_LED = 65
        self.ADDR_GOAL_CURRENT = 102
        self.ADDR_GOAL_POSITION = 116
        self.ADDR_PRESENT_POSITION = 132

        # Gripper Hardware Settings (Feetech STS ID 21 on Protocol 1.0)
        self.GRIPPER_ID = 21
        self.GRIPPER_OPEN = 3865      # 0% Squeeze (Open)
        self.GRIPPER_CLOSED = 2800    # 100% Squeeze (Closed)

        # 2% Micro-Precision Trigger Filter
        self.GRIP_ALPHA = 0.50
        self.smoothed_squeeze_pct = 0.0
        self.TRIGGER_TRAVEL_TICKS = 240.0

        # Bilateral Haptics
        self.HAPTIC_INTENSITY = 0.70
        self.LOAD_DEADBAND = 85
        self.MAX_FEEDBACK_CURRENT = 280
        self.current_haptic_state = False

        # Arm Motion Filter Parameters (50 Hz control loop)
        self.ARM_ALPHA = 0.38
        self.DEADBAND_DEG = 0.08
        self.MAX_VELOCITY_DEG = 8.0

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

        self.trigger_idle_ticks = 1964
        self.last_sent_gripper_reg = None

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
    # MASTER HARDWARE METHODS
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

    def set_master_trigger_torque(self, enable=False):
        val = 1 if enable else 0
        try:
            self.master_pkh.write1ByteTxRx(
                self.master_ph, int(self.trigger_id), int(self.ADDR_TORQUE_ENABLE), val
            )
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
                pos, res, _ = self.master_pkh.read4ByteTxRx(self.master_ph, int(dxl_id), int(self.ADDR_PRESENT_POSITION))
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
            pos, res, _ = self.master_pkh.read4ByteTxRx(self.master_ph, int(self.trigger_id), int(self.ADDR_PRESENT_POSITION))
            if res != 0:
                return self.trigger_idle_ticks
            pos = int(pos) & 0xFFFFFFFF
            if pos > 0x7FFFFFFF:
                pos -= 0x100000000
            return pos
        except Exception:
            return self.trigger_idle_ticks

    def apply_trigger_haptics(self, slave_load):
        load_mag = slave_load & 0x03FF
        if load_mag < self.LOAD_DEADBAND:
            if self.current_haptic_state:
                self.set_master_trigger_torque(enable=False)
                self.current_haptic_state = False
            return 0

        excess_load = load_mag - self.LOAD_DEADBAND
        target_current = int(excess_load * self.HAPTIC_INTENSITY)
        target_current = min(self.MAX_FEEDBACK_CURRENT, max(30, target_current))

        if not self.current_haptic_state:
            self.set_master_trigger_torque(enable=True)
            self.current_haptic_state = True

        self.master_pkh.write2ByteTxRx(self.master_ph, int(self.trigger_id), int(self.ADDR_GOAL_CURRENT), target_current)
        self.master_pkh.write4ByteTxRx(self.master_ph, int(self.trigger_id), int(self.ADDR_GOAL_POSITION), int(self.trigger_idle_ticks))
        return target_current

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
                key1 = (int(rx[0]) >> 4) & 0x0F
                key2 = int(rx[0]) & 0x0F
                prox = int(rx[2])
                return (key1 == 0), (key2 == 0), (prox == 0)
        except Exception:
            pass
        return False, False, False

    # ------------------------------------------------------------
    # 2% MICRO-STEPPING GRIPPER CONTROL
    # ------------------------------------------------------------
    def read_gripper_pos(self):
        pos, res, _ = self.grip_pkh.read2ByteTxRx(self.grip_ph, self.GRIPPER_ID, 56)
        return pos if res == 0 else None

    def read_gripper_load(self):
        load, res, _ = self.grip_pkh.read2ByteTxRx(self.grip_ph, self.GRIPPER_ID, 60)
        return load if res == 0 else 0

    def configure_microstep_gripper(self, enable=True):
        if not self.grip_ph:
            return
        try:
            if enable:
                self.grip_pkh.write1ByteTxRx(self.grip_ph, self.GRIPPER_ID, 55, 0)
                time.sleep(0.02)
                self.grip_pkh.write1ByteTxRx(self.grip_ph, self.GRIPPER_ID, 33, 0)
                self.grip_pkh.write2ByteTxRx(self.grip_ph, self.GRIPPER_ID, 9, 0)
                self.grip_pkh.write2ByteTxRx(self.grip_ph, self.GRIPPER_ID, 11, 4095)

                # Calibrated 2% Micro-Step Registers
                self.grip_pkh.write1ByteTxRx(self.grip_ph, self.GRIPPER_ID, 21, 36)  # P Gain
                self.grip_pkh.write1ByteTxRx(self.grip_ph, self.GRIPPER_ID, 22, 14)  # D Gain
                self.grip_pkh.write1ByteTxRx(self.grip_ph, self.GRIPPER_ID, 23, 2)   # I Gain
                self.grip_pkh.write1ByteTxRx(self.grip_ph, self.GRIPPER_ID, 24, 1)   # 1-Tick Deadband

                self.grip_pkh.write2ByteTxRx(self.grip_ph, self.GRIPPER_ID, 48, 950)  # Max Torque Limit
                self.grip_pkh.write2ByteTxRx(self.grip_ph, self.GRIPPER_ID, 44, 2600) # Speed
                self.grip_pkh.write1ByteTxRx(self.grip_ph, self.GRIPPER_ID, 41, 70)   # Accel Profile
                self.grip_pkh.write1ByteTxRx(self.grip_ph, self.GRIPPER_ID, 40, 1)    # Torque ON
                time.sleep(0.05)
            else:
                self.grip_pkh.write1ByteTxRx(self.grip_ph, self.GRIPPER_ID, 40, 0)
                self.grip_pkh.write2ByteTxRx(self.grip_ph, self.GRIPPER_ID, 48, 0)
        except Exception:
            pass

    def move_percentage_microstep(self, raw_pct, blocking=False):
        raw_pct = float(np.clip(raw_pct, 0.0, 1.0))
        self.smoothed_squeeze_pct = self.GRIP_ALPHA * raw_pct + (1.0 - self.GRIP_ALPHA) * self.smoothed_squeeze_pct
        target = int(round(self.GRIPPER_OPEN - self.smoothed_squeeze_pct * (self.GRIPPER_OPEN - self.GRIPPER_CLOSED)))

        if not blocking:
            if self.last_sent_gripper_reg is None or abs(target - self.last_sent_gripper_reg) >= 1:
                self.grip_pkh.write2ByteTxRx(self.grip_ph, self.GRIPPER_ID, 42, target)
                self.last_sent_gripper_reg = target
            return target

        self.grip_pkh.write2ByteTxRx(self.grip_ph, self.GRIPPER_ID, 42, target)
        self.last_sent_gripper_reg = target
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
        print(" [STEP 1] INITIALIZING HARDWARE & MOVING ROBOT ARM")
        print("==================================================")

        print(f"[+] Connecting Master Joystick on {self.master_port}...")
        self.master_ph = PortHandler(self.master_port)
        self.master_pkh = PacketHandler(2.0)
        if not self.master_ph.openPort() or not self.master_ph.setBaudRate(self.master_baudrate):
            raise RuntimeError(f"Could not open {self.master_port}")

        self.set_master_arm_torque(enable=False)
        self.set_master_trigger_torque(enable=False)
        self.set_leds(enable=False)

        print(f"[+] Connecting Gripper on {self.gripper_port}...")
        self.grip_ph = PortHandler(self.gripper_port)
        self.grip_pkh = PacketHandler(1.0)
        if not self.grip_ph.openPort() or not self.grip_ph.setBaudRate(self.gripper_baudrate):
            raise RuntimeError(f"Could not open {self.gripper_port}")

        print(f"\n[+] Connecting Right Nova 2 at {self.robot_ip}:29999...")
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

        print(f"[+] Moving Right Arm to Initial Target Pose: {self.home_pose}...")
        r_str = ",".join(f"{x:.4f}" for x in self.home_pose)
        self._send_cmd_sync(f"JointMovJ({r_str})")
        time.sleep(4.0)
        print("[+] Robot arm arrived and locked at initial pose.")

    # ============================================================
    # STEP 2: APPLY 2% COMPLIANCE & OPEN GRIPPER TO 3865 TICKS
    # ============================================================
    def step2_open_gripper_to_max(self):
        print("\n==================================================")
        print(" [STEP 2] 2% GAINS & OPENING GRIPPER TO 3865      ")
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
        print(">>> Grasp the Right handle (keep trigger relaxed): Countdown runs when held.\n")

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

        self.set_leds(enable=True)
        print(f"\n\n\033[92m[+] CALIBRATION COMPLETE! TRIGGER IDLE BASELINE: {self.trigger_idle_ticks}\033[0m\n")

    # ============================================================
    # STEP 4: LIVE 50 Hz TELEOP WITH 2% PRECISION & HAPTICS
    # ============================================================
    def step4_live_teleop(self):
        print("==================================================")
        print(">>> LIVE RIGHT ARM + 2% MICRO-PRECISION TELEOP (50 Hz)")
        print(f"    - Resolution      : ~21.3 ticks per 2% squeeze")
        print(f"    - Trigger Idle    : OPEN  ({self.GRIPPER_OPEN} ticks)")
        print(f"    - Trigger Squeeze : CLOSE ({self.GRIPPER_CLOSED} ticks)")
        print("    - Yellow Button   : [REPOSITION CLUTCH] -> Cuts Torque on Arm & Gripper")
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
                        self.set_master_arm_torque(enable=False)
                        self.set_leds(enable=True)
                        time.sleep(0.02)
                        self.locked_ticks = self.read_encoders_safe(self.locked_ticks)
                        self.baseline_robot = np.copy(self.smoothed_joints)
                        prev_delta = np.zeros(6, dtype=np.float64)
                        self.state = "RUNNING"
                    else:
                        print("\n[!] REPOSITION MODE: All Torques Released (Free-Hand Adjust)")
                        self.state = "REPOSITION"
                        self.configure_microstep_gripper(enable=False)
                        self.set_master_arm_torque(enable=False)
                        self.set_master_trigger_torque(enable=False)
                        self.set_leds(enable=False)

                # --- PROXIMITY RE-SYNC ---
                if prox and not was_holding:
                    self.locked_ticks = self.read_encoders_safe(self.locked_ticks)
                    self.baseline_robot = np.copy(self.smoothed_joints)
                    prev_delta = np.zeros(6, dtype=np.float64)
                was_holding = prox

                # --- 1. SUB-TICK PROPORTIONAL TRIGGER READING ---
                trig_now = self.read_trigger_safe()
                displacement = abs(trig_now - self.trigger_idle_ticks)
                active_pull = max(0, displacement - 6)  # 6-tick micro deadband
                squeeze_pct = np.clip(active_pull / self.TRIGGER_TRAVEL_TICKS, 0.0, 1.0)

                target_pos = self.GRIPPER_OPEN
                if self.state == "RUNNING":
                    target_pos = self.move_percentage_microstep(squeeze_pct, blocking=False)

                # --- 2. BILATERAL HAPTICS ---
                slave_load = self.read_gripper_load()
                haptic_force_mA = 0
                if self.state == "RUNNING":
                    haptic_force_mA = self.apply_trigger_haptics(slave_load)

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
            self.set_master_trigger_torque(enable=False)
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
    app = RightArmAndPrecisionGripperTeleop(
        robot_ip="192.168.5.2",
        master_port="COM3",
        gripper_port="COM7",
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