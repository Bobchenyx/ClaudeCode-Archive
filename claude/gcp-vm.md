

## NVMe for GCP-A100 VM

```
lsblk | grep nvme
```

```bash
#!/bin/bash
apt-get install -y mdadm
DEVICES=$(ls /dev/nvme0n* 2>/dev/null)
COUNT=$(echo $DEVICES | wc -w)
if [ $COUNT -gt 0 ] && [ ! -e /dev/md0 ]; then
  mdadm --create /dev/md0 --level=0 --raid-devices=$COUNT $DEVICES
  mkfs.ext4 -F /dev/md0
  mkdir -p /mnt/nvme
  mount /dev/md0 /mnt/nvme
  chmod 777 /mnt/nvme
  echo '/dev/md0 /mnt/nvme ext4 defaults 0 0' >> /etc/fstab
fi
```
