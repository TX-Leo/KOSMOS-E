# !!!!!make sure the ubuntu version is 20.04

 

wget https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb

sudo dpkg -i packages-microsoft-prod.deb

sudo apt-get update

sudo apt-get install blobfuse=1.3.2 -y

 

CN=zhiwang

AN=msranlpintern

AK=TDiIfKsHfxN+/5CFjVoWvWT3s3feQWnbhmp8DH3coHZEZB1FRMIDaaC8ijOfLA+An4RB/fUNdpUXCjBgX64vdQ==
 # you should change to your password instead of using mine, or you will use my datablob

 

MOUNT_DIR=/mnt/${AN}

 

CFG_PATH=~/fuse_connection_${CN}.cfg

MTP=/mnt/localdata/blobfusetmp_temp_${CN}

 

sudo mkdir -p ${MTP}

sudo chmod 777 ${MTP}

 

printf 'accountName %s\naccountKey %s\ncontainerName %s\n' ${AN} ${AK} ${CN} > ${CFG_PATH}

 

sudo chmod 600 ${CFG_PATH}

 

sudo mkdir -p ${MOUNT_DIR}

sudo chmod 777 ${MOUNT_DIR}

blobfuse ${MOUNT_DIR} --tmp-path=${MTP}  --config-file=${CFG_PATH} -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120


CN=unilm

AN=conversationhub

AK=Eb/MuazMyTqDj/WS+/020UKkGLiff/iI6go4Obqp6o/dQ7rUtT7THisY5y9o5f3ch5kGVdjvc2E5+ASt2zLXSg==
 # you should change to your password instead of using mine, or you will use my datablob

 

MOUNT_DIR=/mnt/${AN}

 

CFG_PATH=~/fuse_connection_${CN}.cfg

MTP=/mnt/localdata/blobfusetmp_temp_${CN}

 

sudo mkdir -p ${MTP}

sudo chmod 777 ${MTP}

 

printf 'accountName %s\naccountKey %s\ncontainerName %s\n' ${AN} ${AK} ${CN} > ${CFG_PATH}

 

sudo chmod 600 ${CFG_PATH}

 

sudo mkdir -p ${MOUNT_DIR}

sudo chmod 777 ${MOUNT_DIR}

blobfuse ${MOUNT_DIR} --tmp-path=${MTP}  --config-file=${CFG_PATH} -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120


CN=unilm

AN=msranlp

AK=OnSdaQJfGJ9YJ7tvFKNg9KkiQqXv1RHcc81oT5rxXmNzxWKTVF9vOmpn7+iIh44HoYG+h1pJFHNpIyrmwp2J0w==
 # you should change to your password instead of using mine, or you will use my datablob

 

MOUNT_DIR=/mnt/${AN}

 

CFG_PATH=~/fuse_connection_${CN}.cfg

MTP=/mnt/localdata/blobfusetmp_temp_${CN}

 

sudo mkdir -p ${MTP}

sudo chmod 777 ${MTP}

 

printf 'accountName %s\naccountKey %s\ncontainerName %s\n' ${AN} ${AK} ${CN} > ${CFG_PATH}

 

sudo chmod 600 ${CFG_PATH}

 

sudo mkdir -p ${MOUNT_DIR}

sudo chmod 777 ${MOUNT_DIR}

blobfuse ${MOUNT_DIR} --tmp-path=${MTP}  --config-file=${CFG_PATH} -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120