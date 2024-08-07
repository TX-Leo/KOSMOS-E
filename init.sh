#!/bin/bash

set -e 

apt update

apt install sudo -y

bash blob.sh
bash vl_setup_xl.sh