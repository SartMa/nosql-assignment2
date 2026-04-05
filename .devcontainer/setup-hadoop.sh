#!/bin/bash
sudo apt-get update && sudo apt-get install -y wget curl
if [ ! -d "/opt/hadoop" ]; then
  wget https://archive.apache.org/dist/hadoop/common/hadoop-3.3.6/hadoop-3.3.6.tar.gz -O /tmp/hadoop.tar.gz
  sudo tar -xzf /tmp/hadoop.tar.gz -C /opt
  sudo mv /opt/hadoop-3.3.6 /opt/hadoop
  sudo rm /tmp/hadoop.tar.gz
fi

# Add environment variables to the interactive shell profile
echo 'export HADOOP_HOME=/opt/hadoop' >> ~/.bashrc
echo 'export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin' >> ~/.bashrc
echo 'export HADOOP_CLASSPATH=$(pwd)/opennlp-tools-1.9.3.jar' >> ~/.bashrc
