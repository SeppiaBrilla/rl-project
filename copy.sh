#!/bin/bash

# The command you want to run
COMMAND=" scp -P 2625 ucloud@ssh.cloud.sdu.dk:/home/ucloud/rl-project/car_82* ./"

while true; do
    # Execute the command
    eval $COMMAND
    
    # Wait for 300 seconds (5 minutes)
    sleep 300
done