import os
import sys
import time
import msvcrt
import numpy as np

SDK_PATH = r"C:\Users\Rohit\Downloads\dobot_xtrainer_0813\dobot_xtrainer-master"
if SDK_PATH not in sys.path:
    sys.path.insert(0, SDK_PATH)

from dynamixel_sdk import PortHandler, PacketHandler

os.system("")

# ------------------------------------------------------------
# HARDWARE CONFIGURATION
# ------------------------------------------------------------
MASTER_PORT = "COM5"
GRIPPER_PORT = "COM6"
MASTER_BAUDRATE = 2000000
GRIPPER_BAUDRATE = 1000000

TRIGGER_ID = 8     # Left Joystick Trigger (Protocol 2.0)
GRIPPER_ID = 22    # Left Slave Gripper (Protocol 1.0)
BROADCAST_ID = 254 # Universal Feetech Broadcast ID

# CALIBRATED OPERATING RANGE
GRIPPER_OPEN = 2000       # 0% Squeeze
GRIPPER_CLOSED = 980      # 100% Squeeze
TOTAL_STROKE = GRIPPER_OPEN - GRIPPER_CLOSED

# SMOOTH HAPTIC TUNING
HAPTIC_INTENSITY = 0.65       # Force gain
LOAD_DEADBAND = 40            # Gripper load threshold
MAX_FEEDBACK_CURRENT = 260    # Peak current limit (mA)
HAPTIC_SMOOTH_ALPHA = 0.25    # Low-pass filter (smooths out motor cogging)

# TRIGGER ENCODER MAPPING
PRECISION_BUCKETS = 200.0
GRIP_ALPHA = 0.65
TRIGGER_TRAVEL_TICKS = 240.0
TRIGGER_DEADBAND = 4.0
EXP_CURVE_GAMMA = 1.35

# Protocol 2.0 Registers (Master Trigger)
ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_LED = 65
ADDR_GOAL_CURRENT = 102
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132

# State variables
last_known_gripper_pos = GRIPPER_OPEN
last_known_gripper_load = 0
filtered_haptic_current = 0.0
last_sent_current = None

print("=" * 65)
print(" CONTINUOUS COMPLIANT HAPTIC TELEOP (MODE 5: CURRENT-POSITION)")
print(f" Range: {GRIPPER_OPEN} -> {GRIPPER_CLOSED} ticks")
print(" [Yellow Button] or [T] Key: Instant Torque Release / Re-engage")
print("=" * 65)

master_ph = PortHandler(MASTER_PORT)
master_pkh = PacketHandler(2.0)
if not master_ph.openPort() or not master_ph.setBaudRate(MASTER_BAUDRATE):
    print(f"[-] Failed to open {MASTER_PORT}.")
    sys.exit(1)

grip_ph = PortHandler(GRIPPER_PORT)
grip_pkh = PacketHandler(1.0)
if not grip_ph.openPort() or not grip_ph.setBaudRate(GRIPPER_BAUDRATE):
    print(f"[-] Failed to open {GRIPPER_PORT}.")
    master_ph.closePort()
    sys.exit(1)


def setup_master_haptic_mode():
    """Sets master trigger to Operating Mode 5 (Current-based Position Mode) for smooth spring resistance."""
    try:
        master_pkh.write1ByteTxRx(master_ph, TRIGGER_ID, ADDR_TORQUE_ENABLE, 0)
        time.sleep(0.02)
        master_pkh.write1ByteTxRx(master_ph, TRIGGER_ID, ADDR_OPERATING_MODE, 5)  # Mode 5
        master_pkh.write1ByteTxRx(master_ph, TRIGGER_ID, ADDR_TORQUE_ENABLE, 1)  # Keep torque engaged continuously
        time.sleep(0.02)
    except Exception:
        pass


def read_trigger(idle_fallback=2048):
    try:
        pos, res, _ = master_pkh.read4ByteTxRx(master_ph, TRIGGER_ID, ADDR_PRESENT_POSITION)
        if res != 0:
            return idle_fallback
        pos = int(pos) & 0xFFFFFFFF
        if pos > 0x7FFFFFFF:
            pos -= 0x100000000
        return pos
    except Exception:
        return idle_fallback


def read_gripper_pos():
    global last_known_gripper_pos
    try:
        pos, res, _ = grip_pkh.read2ByteTxRx(grip_ph, GRIPPER_ID, 56)
        if res == 0 and pos is not None:
            last_known_gripper_pos = pos
            return pos
    except Exception:
        pass
    return last_known_gripper_pos


def read_gripper_load():
    global last_known_gripper_load
    try:
        load, res, _ = grip_pkh.read2ByteTxRx(grip_ph, GRIPPER_ID, 60)
        if res == 0 and load is not None:
            last_known_gripper_load = load & 0x03FF
            return last_known_gripper_load
    except Exception:
        pass
    return last_known_gripper_load


def apply_smooth_haptics(slave_load, idle_ticks):
    """Calculates continuous, low-pass filtered spring resistance."""
    global filtered_haptic_current, last_sent_current
    load_mag = slave_load & 0x03FF

    if load_mag > LOAD_DEADBAND:
        excess_load = load_mag - LOAD_DEADBAND
        raw_target_current = float(excess_load * HAPTIC_INTENSITY)
        raw_target_current = min(float(MAX_FEEDBACK_CURRENT), raw_target_current)
    else:
        # Transparent zero-resistance baseline
        raw_target_current = 0.0

    # Low-pass filter to eliminate current stepping spikes
    filtered_haptic_current = (
        HAPTIC_SMOOTH_ALPHA * raw_target_current + (1.0 - HAPTIC_SMOOTH_ALPHA) * filtered_haptic_current
    )
    target_int = int(round(filtered_haptic_current))

    # Send update only if value shifts by >= 2 mA
    if last_sent_current is None or abs(target_int - last_sent_current) >= 2:
        try:
            # Maintain idle homing target while dynamically scaling holding current
            master_pkh.write4ByteTxRx(master_ph, TRIGGER_ID, ADDR_GOAL_POSITION, int(idle_ticks))
            master_pkh.write2ByteTxRx(master_ph, TRIGGER_ID, ADDR_GOAL_CURRENT, target_int)
            last_sent_current = target_int
        except Exception:
            pass

    return target_int


def get_official_keys():
    try:
        master_ph.clearPort()
        master_ph.writePort([0xAA, 0x55, 0xAA])
        rx = []
        t0 = time.time()
        while len(rx) < 4 and (time.time() - t0) < 0.02:
            chunk = master_ph.readPort(int(4 - len(rx)))
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


def force_kill_torque():
    try:
        master_pkh.write1ByteTxRx(master_ph, TRIGGER_ID, ADDR_TORQUE_ENABLE, 0)
    except Exception:
        pass

    for t_id in [GRIPPER_ID, BROADCAST_ID]:
        try:
            grip_pkh.write1ByteTxRx(grip_ph, t_id, 55, 0)
            grip_pkh.write1ByteTxRx(grip_ph, t_id, 40, 0)
            grip_pkh.write2ByteTxRx(grip_ph, t_id, 48, 0)
            grip_pkh.write2ByteTxRx(grip_ph, t_id, 44, 0)
            grip_pkh.write2ByteTxRx(grip_ph, t_id, 42, 0)
            grip_pkh.write1ByteTxRx(grip_ph, t_id, 55, 1)
        except Exception:
            pass


def enable_gripper_torque():
    try:
        grip_pkh.write1ByteTxRx(grip_ph, GRIPPER_ID, 55, 0)
        time.sleep(0.02)
        grip_pkh.write1ByteTxRx(grip_ph, GRIPPER_ID, 33, 0)
        grip_pkh.write2ByteTxRx(grip_ph, GRIPPER_ID, 9, 0)
        grip_pkh.write2ByteTxRx(grip_ph, GRIPPER_ID, 11, 4095)

        grip_pkh.write1ByteTxRx(grip_ph, GRIPPER_ID, 21, 52)
        grip_pkh.write1ByteTxRx(grip_ph, GRIPPER_ID, 22, 20)
        grip_pkh.write1ByteTxRx(grip_ph, GRIPPER_ID, 23, 4)
        grip_pkh.write1ByteTxRx(grip_ph, GRIPPER_ID, 24, 0)

        grip_pkh.write2ByteTxRx(grip_ph, GRIPPER_ID, 48, 850)
        grip_pkh.write2ByteTxRx(grip_ph, GRIPPER_ID, 44, 3200)
        grip_pkh.write1ByteTxRx(grip_ph, GRIPPER_ID, 41, 100)
        grip_pkh.write1ByteTxRx(grip_ph, GRIPPER_ID, 40, 1)
        time.sleep(0.02)
    except Exception:
        pass


try:
    for arm_id in [1, 2, 3, 4, 5, 6, 7]:
        master_pkh.write1ByteTxRx(master_ph, arm_id, ADDR_TORQUE_ENABLE, 0)
    master_pkh.write1ByteTxRx(master_ph, 8, ADDR_LED, 1)

    time.sleep(0.2)
    trigger_idle = read_trigger(2048)
    print(f"[+] Trigger Baseline: {trigger_idle} ticks")

    # Set up smooth continuous haptics mode
    setup_master_haptic_mode()

    print(f"[+] Initializing & Homing to OPEN ({GRIPPER_OPEN} ticks)...")
    enable_gripper_torque()
    try:
        grip_pkh.write2ByteTxRx(grip_ph, GRIPPER_ID, 42, GRIPPER_OPEN)
    except Exception:
        pass
    time.sleep(1.0)
    print(f"[+] Gripper Ready at: {read_gripper_pos()} ticks")

    print("\n>>> LIVE TELEOPERATION STREAMING\n")

    smoothed_pct = 0.0
    last_sent_target = None
    torque_released = False
    last_btn_time = 0.0
    DT = 0.02

    while True:
        t0 = time.time()
        now = time.time()

        key_triggered = False
        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key in [b"t", b"T", b" "]:
                key_triggered = True

        yel, _, _ = get_official_keys()

        if (key_triggered or yel) and (now - last_btn_time > 0.35):
            last_btn_time = now
            torque_released = not torque_released
            if torque_released:
                force_kill_torque()
                master_pkh.write1ByteTxRx(master_ph, 8, ADDR_LED, 0)
                last_sent_target = None
                print("\n\033[93m>>> [EVENT] TORQUE RELEASED (MOTOR LIMP)\033[0m")
            else:
                setup_master_haptic_mode()
                enable_gripper_torque()
                master_pkh.write1ByteTxRx(master_ph, 8, ADDR_LED, 1)
                last_sent_target = None
                print("\n\033[92m>>> [EVENT] TORQUE RE-ENGAGED\033[0m")

        if not torque_released:
            trig_now = read_trigger(trigger_idle)
            displacement = abs(trig_now - trigger_idle)
            active_pull = max(0.0, displacement - TRIGGER_DEADBAND)
            linear_pct = float(np.clip(active_pull / TRIGGER_TRAVEL_TICKS, 0.0, 1.0))

            curved_pct = float(np.power(linear_pct, EXP_CURVE_GAMMA))
            quantized_pct = np.round(curved_pct * PRECISION_BUCKETS) / PRECISION_BUCKETS
            smoothed_pct = GRIP_ALPHA * quantized_pct + (1.0 - GRIP_ALPHA) * smoothed_pct

            target = int(round(GRIPPER_OPEN - smoothed_pct * TOTAL_STROKE))

            if last_sent_target is None or abs(target - last_sent_target) >= 1:
                try:
                    grip_pkh.write2ByteTxRx(grip_ph, GRIPPER_ID, 42, target)
                    last_sent_target = target
                except Exception:
                    pass

            curr_load = read_gripper_load()
            haptic_force_mA = apply_smooth_haptics(curr_load, trigger_idle)

            status_tag = "\033[92m[ACTIVE]\033[0m"
            target_str = f"{target:4d}"
        else:
            status_tag = "\033[93m[TORQUE OFF]\033[0m"
            target_str = " OFF"
            curr_load = 0
            haptic_force_mA = 0

        curr_pos = read_gripper_pos()
        pct_display = float(smoothed_pct * 100.0)
        haptic_str = f"\033[96m{haptic_force_mA:3d} mA\033[0m" if haptic_force_mA > 0 else "  0 mA"

        sys.stdout.write(
            f"\r\033[KSTATUS: {status_tag:<26} | SQUEEZE: {pct_display:5.1f}% | "
            f"TARGET: {target_str} | POS: {curr_pos:4d} | LOAD: {curr_load:3d} | HAPTIC: {haptic_str:<15}"
        )
        sys.stdout.flush()

        elapsed = time.time() - t0
        if elapsed < DT:
            time.sleep(DT - elapsed)

except KeyboardInterrupt:
    print("\n\n[-] Teleoperation stopped by user.")

finally:
    print("[+] Releasing all torques and closing ports...")
    try:
        master_pkh.write1ByteTxRx(master_ph, 8, ADDR_LED, 0)
        for arm_id in [1, 2, 3, 4, 5, 6, 7, 8]:
            master_pkh.write1ByteTxRx(master_ph, arm_id, ADDR_TORQUE_ENABLE, 0)
        force_kill_torque()
    except Exception:
        pass

    try:
        master_ph.closePort()
        grip_ph.closePort()
    except Exception:
        pass
    print("[+] Torque 100% released. System safely offline.")