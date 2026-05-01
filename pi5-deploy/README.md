# Pi 5 Deployment Staging

Holds Hailo binaries that will be transferred to the Raspberry Pi 5 + AI HAT+ 2.
**Files are gitignored** — they have license restrictions on redistribution.

## Files

| File | Source | Pi 5 install |
|------|--------|--------------|
| `hailort-pcie-driver_*.deb` | Hailo Dev Zone | `sudo dpkg -i hailort-pcie-driver_*.deb` |
| `hailort_*_arm64.deb` | Hailo Dev Zone (필요 시) | `sudo dpkg -i hailort_*_arm64.deb` |

## Pi 5 install order (manual, after `setup.sh` if APT fails)

```bash
# 1. Copy this folder to Pi 5
scp -r pi5-deploy/ pi@raspberrypi.local:~/

# 2. SSH into Pi 5
ssh pi@raspberrypi.local

# 3. Install
cd ~/pi5-deploy
sudo apt install -y ./hailort-pcie-driver_*.deb
# Reboot Pi 5
sudo reboot

# 4. After reboot, verify
lspci | grep -i hailo
hailortcli fw-control identify
```

## Notes

- The default path in `setup.sh` is to install via Hailo APT repo. This staging folder is the **manual fallback** if APT is unreachable on the Pi.
- The PCIe driver version must match the HailoRT version installed on the Pi. If you mix-and-match, `hailortcli` will refuse to talk to the device.
