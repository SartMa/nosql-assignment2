#!/bin/bash
sudo apt-get update && sudo apt-get install -y wget curl
if [ ! -d "/opt/hadoop" ]; then
  wget https://archive.apache.org/dist/hadoop/common/hadoop-3.3.6/hadoop-3.3.6.tar.gz -O /tmp/hadoop.tar.gz
  sudo tar -xzf /tmp/hadoop.tar.gz -C /opt
  sudo mv /opt/hadoop-3.3.6 /opt/hadoop
  rm /tmp/hadoop.tar.gz
fi
