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

os.system("")  # Enable ANSI terminal formatting on Windows


class XTrainerBimanualLEDTeleop:
    def __init__(
        self,
        left_robot_ip="192.168.5.1",
        left_master_port="COM5",
        right_robot_ip="192.168.5.2",
        right_master_port="COM3",
        master_baudrate=2000000,
    ):
        self.left_robot_ip = left_robot_ip
        self.left_master_port = left_master_port
        self.right_robot_ip = right_robot_ip
        self.right_master_port = right_master_port
        self.master_baudrate = int(master_baudrate)

        # Initial Target Robot Poses (Degrees)
        self.left_home_pose = [-89.7758, 1.3375, -86.5728, -2.3068, 87.0675, -0.4531]
        self.right_home_pose = [89.7758, 1.3375, 86.5728, 2.3068, -87.0675, 0.4531]

        # 6-DOF Active Joint IDs
        self.left_joint_ids = [1, 2, 4, 5, 6, 7]
        self.left_all_ids = [1, 2, 3, 4, 5, 6, 7, 8]

        self.right_joint_ids = [11, 12, 14, 15, 16, 17]
        self.right_all_ids = [11, 12, 13, 14, 15, 16, 17, 18]

        # Kinematic Direction Signs (J1 Inverted on Left for Intuitive Push/Pull)
        self.left_joint_signs = np.array([1, 1, -1, -1, -1, 1], dtype=np.float64)
        self.right_joint_signs = np.array([1, 1, -1, -1, -1, 1], dtype=np.float64)

        # Dynamixel Protocol 2.0 Registers
        self.ADDR_TORQUE_ENABLE = 64
        self.ADDR_LED = 65            # 1 = LED ON, 0 = LED OFF
        self.ADDR_PRESENT_POSITION = 132

        # High-Speed Dynamic Filter Parameters
        self.ALPHA = 0.38             # High responsiveness
        self.DEADBAND_DEG = 0.10      # Vibration rejection
        self.MAX_VELOCITY_DEG = 8.0   # Velocity clamp per frame (400°/sec max)

        # Hardware Handlers
        self.left_port_handler = None
        self.left_packet_handler = None
        self.right_port_handler = None
        self.right_packet_handler = None
        self.left_sock = None
        self.right_sock = None

        # State Vectors
        self.left_baseline_robot = np.array(self.left_home_pose, dtype=np.float64)
        self.right_baseline_robot = np.array(self.right_home_pose, dtype=np.float64)
        self.left_locked_ticks = None
        self.right_locked_ticks = None
        self.left_smoothed_joints = np.array(self.left_home_pose, dtype=np.float64)
        self.right_smoothed_joints = np.array(self.right_home_pose, dtype=np.float64)

        # Independent Arm Macro States
        self.left_state = "RUNNING"
        self.right_state = "RUNNING"
        self.left_last_btn_time = 0.0
        self.right_last_btn_time = 0.0

    # ------------------------------------------------------------
    # LED CONTROL METHODS
    # ------------------------------------------------------------
    def set_leds(self, side="both", enable=True):
        """Controls the onboard status LEDs on all master servos."""
        val = 1 if enable else 0
        try:
            if side in ("left", "both") and self.left_packet_handler and self.left_port_handler:
                for dxl_id in self.left_all_ids:
                    self.left_packet_handler.write1ByteTxRx(
                        self.left_port_handler, int(dxl_id), int(self.ADDR_LED), val
                    )
            if side in ("right", "both") and self.right_packet_handler and self.right_port_handler:
                for dxl_id in self.right_all_ids:
                    self.right_packet_handler.write1ByteTxRx(
                        self.right_port_handler, int(dxl_id), int(self.ADDR_LED), val
                    )
        except Exception:
            pass

    # ------------------------------------------------------------
    # LOW-LEVEL ROBOT COMMS
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
        """High-rate ServoJ streaming (50 Hz / 20ms)."""
        j_str = ",".join(f"{float(x):.4f}" for x in joint_angles)
        msg = f"ServoJ({j_str},t=0.02)\r\n".encode("utf-8")
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
    # MASTER CONTROLLER METHODS
    # ------------------------------------------------------------
    def set_single_master_torque(self, side="left", enable=False):
        val = 1 if enable else 0
        try:
            if side == "left" and self.left_packet_handler and self.left_port_handler:
                for dxl_id in self.left_all_ids:
                    self.left_packet_handler.write1ByteTxRx(
                        self.left_port_handler, int(dxl_id), int(self.ADDR_TORQUE_ENABLE), val
                    )
            elif side == "right" and self.right_packet_handler and self.right_port_handler:
                for dxl_id in self.right_all_ids:
                    self.right_packet_handler.write1ByteTxRx(
                        self.right_port_handler, int(dxl_id), int(self.ADDR_TORQUE_ENABLE), val
                    )
        except Exception:
            pass

    def read_encoders_safe(self, pkh, ph, joint_ids, fallback_ticks=None):
        """Reads encoder ticks, handles two's complement, and prevents ctypes overflow."""
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

    def get_official_keys(self, ph):
        """Reads Dobot handle buttons & proximity sensor."""
        try:
            ph.clearPort()
            ph.writePort([0xAA, 0x55, 0xAA])

            rx = []
            t0 = time.time()
            while len(rx) < 4 and (time.time() - t0) < 0.02:
                chunk = ph.readPort(int(4 - len(rx)))
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

    # ============================================================
    # STEP 1: INITIALIZE HARDWARE & MOVE BOTH ARMS TO TARGET POSES
    # ============================================================
    def step1_travel_both_robots_to_coordinates(self):
        print("\n==================================================")
        print(" [STEP 1] INITIALIZING HARDWARE & MOVING ROBOTS   ")
        print("==================================================")

        # 1. Connect Left Master (COM5)
        print(f"[+] Connecting Left Master Joystick on {self.left_master_port}...")
        self.left_port_handler = PortHandler(self.left_master_port)
        self.left_packet_handler = PacketHandler(2.0)
        if not self.left_port_handler.openPort() or not self.left_port_handler.setBaudRate(self.master_baudrate):
            raise RuntimeError(f"Could not open {self.left_master_port}")

        # 2. Connect Right Master (COM3)
        print(f"[+] Connecting Right Master Joystick on {self.right_master_port}...")
        self.right_port_handler = PortHandler(self.right_master_port)
        self.right_packet_handler = PacketHandler(2.0)
        if not self.right_port_handler.openPort() or not self.right_port_handler.setBaudRate(self.master_baudrate):
            raise RuntimeError(f"Could not open {self.right_master_port}")

        self.set_single_master_torque("left", enable=False)
        self.set_single_master_torque("right", enable=False)
        self.set_leds("both", enable=False)

        # 3. Connect Left Robot Arm
        print(f"[+] Connecting Left Nova 2 at {self.left_robot_ip}:29999...")
        self.left_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.left_sock.settimeout(5.0)
        self.left_sock.connect((self.left_robot_ip, 29999))

        # 4. Connect Right Robot Arm
        print(f"[+] Connecting Right Nova 2 at {self.right_robot_ip}:29999...")
        self.right_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.right_sock.settimeout(5.0)
        self.right_sock.connect((self.right_robot_ip, 29999))

        # Power On & Enable Both Arms
        for name, sock in [("Left", self.left_sock), ("Right", self.right_sock)]:
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
            self._send_cmd_sync(sock, "SpeedJ(50)")
            self._send_cmd_sync(sock, "AccJ(50)")

        # Move Both Robots to Initial Poses
        print(f"\n[+] Moving Left Robot to: {self.left_home_pose}...")
        l_str = ",".join(f"{x:.4f}" for x in self.left_home_pose)
        self._send_cmd_sync(self.left_sock, f"JointMovJ({l_str})")

        print(f"[+] Moving Right Robot to: {self.right_home_pose}...")
        r_str = ",".join(f"{x:.4f}" for x in self.right_home_pose)
        self._send_cmd_sync(self.right_sock, f"JointMovJ({r_str})")

        time.sleep(4.0)
        print("[+] Both Robots arrived at target home poses.")

    # ============================================================
    # STEP 2: PROXIMITY-GATED ALIGNMENT TIMER (WITH LED SIGNALS)
    # ============================================================
    def step2_reposition_joysticks_with_prox_timer(self, target_seconds=10):
        print("\n==================================================")
        print(f" [STEP 2] JOYSTICKS ALIGNMENT ({target_seconds}s HOLD REQUIRED) ")
        print("==================================================")
        print(">>> Grasp BOTH handles: Timer counts down ONLY when both hands are on handles.\n")

        accumulated_time = 0.0
        last_loop_time = time.time()
        last_blink_time = time.time()
        blink_state = False

        while accumulated_time < target_seconds:
            loop_now = time.time()
            dt = loop_now - last_loop_time
            last_loop_time = loop_now

            _, _, l_prox = self.get_official_keys(self.left_port_handler)
            _, _, r_prox = self.get_official_keys(self.right_port_handler)

            if l_prox and r_prox:
                accumulated_time += dt
                remaining = max(0.0, target_seconds - accumulated_time)
                timer_msg = f"\033[92m[COUNTING DOWN]\033[0m {remaining:04.1f}s left"

                # Blink LEDs while countdown is active
                if loop_now - last_blink_time > 0.3:
                    blink_state = not blink_state
                    self.set_leds("both", enable=blink_state)
                    last_blink_time = loop_now
            else:
                remaining = max(0.0, target_seconds - accumulated_time)
                status_parts = []
                if not l_prox: status_parts.append("HOLD LEFT")
                if not r_prox: status_parts.append("HOLD RIGHT")
                timer_msg = f"\033[93m[PAUSED: {' & '.join(status_parts)}]\033[0m {remaining:04.1f}s"
                self.set_leds("both", enable=False)

            sys.stdout.write(f"\r\033[KTIMER: {timer_msg:<48}")
            sys.stdout.flush()
            time.sleep(0.04)

        # Lock Reference Angles at Countdown Completion
        self.left_locked_ticks = self.read_encoders_safe(self.left_packet_handler, self.left_port_handler, self.left_joint_ids)
        self.right_locked_ticks = self.read_encoders_safe(self.right_packet_handler, self.right_port_handler, self.right_joint_ids)

        self.left_baseline_robot = np.array(self.left_home_pose, dtype=np.float64)
        self.right_baseline_robot = np.array(self.right_home_pose, dtype=np.float64)
        self.left_smoothed_joints = np.copy(self.left_baseline_robot)
        self.right_smoothed_joints = np.copy(self.right_baseline_robot)

        # Turn ON LEDs Solid
        self.set_leds("both", enable=True)
        print("\n\n\033[92m[+] CALIBRATION COMPLETE! SERVO LEDS LIT! TELEOP ENGAGED!\033[0m")

    # ============================================================
    # STEP 3: LIVE HIGH-SPEED BIMANUAL TELEOPERATION
    # ============================================================
    def step3_live_teleop_with_macro(self):
        print("\n==================================================")
        print(">>> LIVE HIGH-SPEED BIMANUAL TELEOPERATION (50 Hz)")
        print("    - LEDs LIT       : Proper calibration confirmed")
        print("    - LEFT JOYSTICK  : Controls LEFT arm only     ")
        print("    - RIGHT JOYSTICK : Controls RIGHT arm only    ")
        print("    - Green Button   : Toggle BRAKE on THAT arm   ")
        print("    - Yellow Button  : Toggle REPOSITION on THAT arm")
        print("==================================================\n")

        DT = 0.02  # 50 Hz control loop (20 ms interval)
        l_prev_delta = np.zeros(6, dtype=np.float64)
        r_prev_delta = np.zeros(6, dtype=np.float64)
        l_was_holding = True
        r_was_holding = True

        try:
            while True:
                t0 = time.time()
                now = time.time()

                l_yel, l_grn, l_prox = self.get_official_keys(self.left_port_handler)
                r_yel, r_grn, r_prox = self.get_official_keys(self.right_port_handler)

                # --- LEFT ARM BUTTON MACROS ---
                if l_grn and (now - self.left_last_btn_time > 0.4):
                    self.left_last_btn_time = now
                    if self.left_state == "BRAKED":
                        self.set_single_master_torque("left", enable=False)
                        self.set_leds("left", enable=True)
                        time.sleep(0.02)
                        self.left_locked_ticks = self.read_encoders_safe(self.left_packet_handler, self.left_port_handler, self.left_joint_ids, self.left_locked_ticks)
                        self.left_baseline_robot = np.copy(self.left_smoothed_joints)
                        l_prev_delta = np.zeros(6, dtype=np.float64)
                        self.left_state = "RUNNING"
                    else:
                        self.left_state = "BRAKED"
                        self.set_single_master_torque("left", enable=True)

                elif l_yel and (now - self.left_last_btn_time > 0.4):
                    self.left_last_btn_time = now
                    if self.left_state == "REPOSITION":
                        time.sleep(0.02)
                        self.left_locked_ticks = self.read_encoders_safe(self.left_packet_handler, self.left_port_handler, self.left_joint_ids, self.left_locked_ticks)
                        self.left_baseline_robot = np.copy(self.left_smoothed_joints)
                        l_prev_delta = np.zeros(6, dtype=np.float64)
                        self.set_leds("left", enable=True)
                        self.left_state = "RUNNING"
                    else:
                        self.left_state = "REPOSITION"
                        self.set_leds("left", enable=False)
                        self.set_single_master_torque("left", enable=False)

                # --- RIGHT ARM BUTTON MACROS ---
                if r_grn and (now - self.right_last_btn_time > 0.4):
                    self.right_last_btn_time = now
                    if self.right_state == "BRAKED":
                        self.set_single_master_torque("right", enable=False)
                        self.set_leds("right", enable=True)
                        time.sleep(0.02)
                        self.right_locked_ticks = self.read_encoders_safe(self.right_packet_handler, self.right_port_handler, self.right_joint_ids, self.right_locked_ticks)
                        self.right_baseline_robot = np.copy(self.right_smoothed_joints)
                        r_prev_delta = np.zeros(6, dtype=np.float64)
                        self.right_state = "RUNNING"
                    else:
                        self.right_state = "BRAKED"
                        self.set_single_master_torque("right", enable=True)

                elif r_yel and (now - self.right_last_btn_time > 0.4):
                    self.right_last_btn_time = now
                    if self.right_state == "REPOSITION":
                        time.sleep(0.02)
                        self.right_locked_ticks = self.read_encoders_safe(self.right_packet_handler, self.right_port_handler, self.right_joint_ids, self.right_locked_ticks)
                        self.right_baseline_robot = np.copy(self.right_smoothed_joints)
                        r_prev_delta = np.zeros(6, dtype=np.float64)
                        self.set_leds("right", enable=True)
                        self.right_state = "RUNNING"
                    else:
                        self.right_state = "REPOSITION"
                        self.set_leds("right", enable=False)
                        self.set_single_master_torque("right", enable=False)

                # --- PROXIMITY RE-SYNC ---
                if l_prox and not l_was_holding:
                    self.left_locked_ticks = self.read_encoders_safe(self.left_packet_handler, self.left_port_handler, self.left_joint_ids, self.left_locked_ticks)
                    self.left_baseline_robot = np.copy(self.left_smoothed_joints)
                    l_prev_delta = np.zeros(6, dtype=np.float64)
                l_was_holding = l_prox

                if r_prox and not r_was_holding:
                    self.right_locked_ticks = self.read_encoders_safe(self.right_packet_handler, self.right_port_handler, self.right_joint_ids, self.right_locked_ticks)
                    self.right_baseline_robot = np.copy(self.right_smoothed_joints)
                    r_prev_delta = np.zeros(6, dtype=np.float64)
                r_was_holding = r_prox

                # --- HIGH-SPEED MOTION STREAMING (50 Hz) ---
                if (self.left_state == "RUNNING") and l_prox:
                    l_ticks = self.read_encoders_safe(self.left_packet_handler, self.left_port_handler, self.left_joint_ids, self.left_locked_ticks)
                    l_raw_delta = (l_ticks - self.left_locked_ticks) * (360.0 / 4096.0) * self.left_joint_signs

                    l_diff = np.abs(l_raw_delta - l_prev_delta)
                    l_act_delta = np.where(l_diff < self.DEADBAND_DEG, l_prev_delta, l_raw_delta)
                    l_prev_delta = l_act_delta

                    l_raw_target = self.left_baseline_robot + l_act_delta
                    l_step = np.clip(l_raw_target - self.left_smoothed_joints, -self.MAX_VELOCITY_DEG, self.MAX_VELOCITY_DEG)
                    self.left_smoothed_joints = self.ALPHA * (self.left_smoothed_joints + l_step) + (1.0 - self.ALPHA) * self.left_smoothed_joints
                    self._send_servoj_fast(self.left_sock, self.left_smoothed_joints)

                if (self.right_state == "RUNNING") and r_prox:
                    r_ticks = self.read_encoders_safe(self.right_packet_handler, self.right_port_handler, self.right_joint_ids, self.right_locked_ticks)
                    r_raw_delta = (r_ticks - self.right_locked_ticks) * (360.0 / 4096.0) * self.right_joint_signs

                    r_diff = np.abs(r_raw_delta - r_prev_delta)
                    r_act_delta = np.where(r_diff < self.DEADBAND_DEG, r_prev_delta, r_raw_delta)
                    r_prev_delta = r_act_delta

                    r_raw_target = self.right_baseline_robot + r_act_delta
                    r_step = np.clip(r_raw_target - self.right_smoothed_joints, -self.MAX_VELOCITY_DEG, self.MAX_VELOCITY_DEG)
                    self.right_smoothed_joints = self.ALPHA * (self.right_smoothed_joints + r_step) + (1.0 - self.ALPHA) * self.right_smoothed_joints
                    self._send_servoj_fast(self.right_sock, self.right_smoothed_joints)

                # --- LIVE STATUS BAR ---
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

                sys.stdout.write(
                    f"\r\033[KLEFT: {l_tag:<18} (J1:{self.left_smoothed_joints[0]:+06.1f}°) | "
                    f"RIGHT: {r_tag:<18} (J1:{self.right_smoothed_joints[0]:+06.1f}°) | "
                    f"LEDS: \033[92m[ON]\033[0m"
                )
                sys.stdout.flush()

                elapsed = time.time() - t0
                if elapsed < DT:
                    time.sleep(DT - elapsed)

        except KeyboardInterrupt:
            print("\n\n[-] Bimanual Teleoperation ended by user (Ctrl+C).")
        finally:
            self.shutdown()

    def shutdown(self):
        print("\n[+] Turning off LEDs, releasing torques, and shutting down cleanly...")

        # Turn Off LEDs & Release Joystick Torques
        try:
            self.set_leds("both", enable=False)
            self.set_single_master_torque("left", enable=False)
            self.set_single_master_torque("right", enable=False)
            if self.left_port_handler:
                self.left_port_handler.closePort()
            if self.right_port_handler:
                self.right_port_handler.closePort()
            print("    -> LEDs off & Master Joysticks closed.")
        except Exception:
            pass

        # Disable Robot Arms
        for name, sock in [("Left", self.left_sock), ("Right", self.right_sock)]:
            try:
                if sock:
                    self._send_cmd_sync(sock, "DisableRobot()")
                    sock.close()
                    print(f"    -> {name} Robot arm disabled & socket closed.")
            except Exception:
                pass

        print("[+] System shutdown complete.")


if __name__ == "__main__":
    app = XTrainerBimanualLEDTeleop(
        left_robot_ip="192.168.5.1",
        left_master_port="COM5",
        right_robot_ip="192.168.5.2",
        right_master_port="COM3",
        master_baudrate=2000000,
    )
    try:
        app.step1_travel_both_robots_to_coordinates()
        app.step2_reposition_joysticks_with_prox_timer(target_seconds=10)
        app.step3_live_teleop_with_macro()
    except Exception as err:
        print(f"\n[-] Execution Error: {err}")
        app.shutdown()