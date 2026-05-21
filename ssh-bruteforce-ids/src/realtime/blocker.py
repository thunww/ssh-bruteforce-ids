from __future__ import annotations

import subprocess


def block_ip_iptables(ip: str) -> None:
    cmd = f"iptables -C INPUT -s {ip} -j DROP || iptables -A INPUT -s {ip} -j DROP"
    subprocess.run(["sudo", "bash", "-lc", cmd], check=False)


def unblock_ip_iptables(ip: str) -> None:
    cmd = f"iptables -D INPUT -s {ip} -j DROP"
    subprocess.run(["sudo", "bash", "-lc", cmd], check=False)