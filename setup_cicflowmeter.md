# Install deps
sudo apt update
sudo apt install openjdk-8-jdk git maven tcpdump

# Clone
git clone https://github.com/ahlashkari/CICFlowMeter.git
cd CICFlowMeter

# Install jnetpcap
cd jnetpcap/linux/jnetpcap-1.4.r1425
sudo mvn install:install-file -Dfile=jnetpcap.jar \
-DgroupId=org.jnetpcap -DartifactId=jnetpcap \
-Dversion=1.4.1 -Dpackaging=jar
cd ../../../..

# Build
chmod +x gradlew
./gradlew clean build
./gradlew installDist

# Capture
sudo tcpdump -i enp2s0 -s 0 -w capture.pcap

# Convert
sudo JAVA_OPTS="-Djava.library.path=jnetpcap/linux/jnetpcap-1.4.r1425" \
build/install/CICFlowMeter/bin/cfm capture.pcap output.csv
