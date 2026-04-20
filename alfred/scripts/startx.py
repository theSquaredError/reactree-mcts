#!/usr/bin/env python


import subprocess
import shlex
import re
import platform
import tempfile
import os
import sys

def pci_records():
    records = []
    command = shlex.split('lspci -vmm')
    output = subprocess.check_output(command).decode()

    for devices in output.strip().split("\n\n"):
        record = {}
        for row in devices.split("\n"):
            if "\t" not in row:
                continue
            key, value = row.split("\t", 1)
            record[key.split(':')[0]] = value.strip()
        if record:
            records.append(record)

    return records

def slot_to_busid(slot):
    parts = re.split(r'[:\.]', slot)
    if len(parts) == 4:
        domain, bus, device, function = parts
    elif len(parts) == 3:
        domain = "0000"
        bus, device, function = parts
    else:
        raise ValueError("Unexpected PCI slot format: %s" % slot)

    domain = int(domain, 16)
    bus = int(bus, 16)
    device = int(device, 16)
    function = int(function, 16)

    if domain != 0:
        return "PCI:%d@%d:%d:%d" % (bus, domain, device, function)
    return "PCI:%d:%d:%d" % (bus, device, function)

def generate_xorg_conf(devices):
    xorg_conf = []

    device_section = """
Section "Device"
    Identifier     "Device{device_id}"
    Driver         "nvidia"
    VendorName     "NVIDIA Corporation"
    BusID          "{bus_id}"
EndSection
"""
    server_layout_section = """
Section "ServerLayout"
    Identifier     "Layout0"
    {screen_records}
EndSection
"""
    screen_section = """
Section "Screen"
    Identifier     "Screen{screen_id}"
    Device         "Device{device_id}"
    DefaultDepth    24
    Option         "AllowEmptyInitialConfiguration" "True"
    SubSection     "Display"
        Depth       24
        Virtual 1024 768
    EndSubSection
EndSection
"""
    screen_records = []
    for i, bus_id in enumerate(devices):
        xorg_conf.append(device_section.format(device_id=i, bus_id=bus_id))
        xorg_conf.append(screen_section.format(device_id=i, screen_id=i))
        screen_records.append('Screen {screen_id} "Screen{screen_id}" 0 0'.format(screen_id=i))

    xorg_conf.append(server_layout_section.format(screen_records="\n    ".join(screen_records)))

    output =  "\n".join(xorg_conf)
    print(output)
    return output

def startx(display):
    if platform.system() != 'Linux':
        raise Exception("Can only run startx on linux")

    devices = []
    for r in pci_records():
        if r.get('Vendor', '') == 'NVIDIA Corporation' \
                and r.get('Class') in ['VGA compatible controller', '3D controller']:
            bus_id = slot_to_busid(r['Slot'])
            devices.append(bus_id)

    if not devices:
        raise Exception("no nvidia cards found")

    try:
        fd, path = tempfile.mkstemp()
        with open(path, "w") as f:
            f.write(generate_xorg_conf(devices))
        command = shlex.split("Xorg -noreset +extension GLX +extension RANDR +extension RENDER -config %s :%s" % (path, display))
        subprocess.call(command)
    finally:
        os.close(fd)
        os.unlink(path)


if __name__ == '__main__':
    display = 0
    if len(sys.argv) > 1:
        display = int(sys.argv[1])
    print("Starting X on DISPLAY=:%s" % display)
    startx(display)