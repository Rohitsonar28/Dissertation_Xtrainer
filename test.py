import os
import sys
import time

SDK_PATH = r"C:\Users\Rohit\Downloads\dobot_xtrainer_0813\dobot_xtrainer-master"
if SDK_PATH not in sys.path:
    sys.path.insert(0, SDK_PATH)

from dynamixel_sdk import PortHandler, PacketHandler

PORT = "COM6"
BAUDRATE = 1000000
PROTOCOL_VERSION = 1.0
GRIPPER_ID = 22

# Register Addresses
ADDR_STS_TORQUE_ENABLE = 40  # STS / Feetech Smart Servo
ADDR_DXL_TORQUE_ENABLE = 24  # Standard Dynamixel
ADDR_STS_PRESENT_POS   = 56  # STS Position
ADDR_DXL_PRESENT_POS   = 36  # Standard Position
ADDR_STS_LOCK          = 55  # EEP-ROM / Parameter Lock

portHandler = PortHandler(PORT)
packetHandler = PacketHandler(PROTOCOL_VERSION)

if not portHandler.openPort() or not portHandler.setBaudRate(BAUDRATE):
    print(f"[-] Could not open {PORT}. Close any other running terminal scripts.")
    sys.exit(1)

print("==================================================")
print("     GRIPPER UNBRAKE & MANUAL POSITION READER     ")
print("==================================================")

try:
    # 1. Clear Parameter Lock and Disable All Holding Torque
    print(f"[+] Releasing holding torque & electronic brake on ID {GRIPPER_ID}...")
    packetHandler.write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_STS_LOCK, 0)
    packetHandler.write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_STS_TORQUE_ENABLE, 0)
    packetHandler.write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_DXL_TORQUE_ENABLE, 0)
    time.sleep(0.1)

    print("✅ BRAKE RELEASED! Motor coils powered off.")
    print(">>> Move the gripper jaws with your hand to test the motion. <<<")
    print(">>> Press Ctrl+C to exit.\n")

    # 2. Live Encoder Stream
    while True:
        pos_sts, res_sts, _ = packetHandler.read2ByteTxRx(portHandler, GRIPPER_ID, ADDR_STS_PRESENT_POS)
        pos_dxl, res_dxl, _ = packetHandler.read2ByteTxRx(portHandler, GRIPPER_ID, ADDR_DXL_PRESENT_POS)
        
        pos = pos_sts if res_sts == 0 else pos_dxl
        print(f"\r  -> Live Gripper Position: {pos:04d} ticks  ", end="", flush=True)
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\n\n[+] Exited.")
finally:
    portHandler.closePort()
    print("[+] Port closed.")