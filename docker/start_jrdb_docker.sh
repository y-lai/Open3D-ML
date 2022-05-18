#!/bin/bash
homedir=/workspace
xhost +local:docker &> /dev/null
usegpu="true"

datasetpath= # path to jrdb dataset root folder here

# assume this script is run at its location
mlpath= $(pwd)/..

if [ $# -eq 0 ]; then
    echo "Using discrete GPU in docker."
    echo "Use -c option if using CPU docker. Note: Tensorflow/Torch in CPU docker won't work since the docker image relies on CUDA."
fi

while getopts "c" OPTION;  do
    case $OPTION in
        c)
            usegpu=""
            ;;
    esac
done

if [ -z $usegpu ]; then
    echo "Using CPU option in docker."
    docker create -it --rm --name jrdb-pointpillar\
        --network host\
        --privileged\
        --mount type=bind,source=$mlpath,target=$homedir/Open3D-ML \
        --mount type=bind,source=$datasetpath,target=$homedir/jrdb \
        ml3d:JRDB
else
    # GPU version - assumes you have nvidia docker installed and configured properly
    docker create -it --rm --name jrdb-pointpillar\
        --gpus all\
        --ipc=host\
        --ulimit memlock=-1\
        --ulimit stack=67108864\
        --network host\
        --privileged\
        --mount type=bind,source=$mlpath,target=$homedir/Open3D-ML \
        --mount type=bind,source=$datasetpath,target=$homedir/jrdb \
        ml3d:JRDB
fi

# add the below into the docker create if visualisations are done in the docker container
# --volume=/tmp/.X11-unix:/tmp/.X11-unix:rw \
# --device=/dev/dri \
# --env=DISPLAY \
# --env=QT_X11_NO_MITSHM=1 \

# copying files across into container
docker cp $(pwd)/container-files/jrdb_pointpillar-preprocess.sh jrdb-pointpillar:/workspace/jrdb_pointpillar-preprocess.sh

docker start jrdb-pointpillar
docker exec -it jrdb-pointpillar bash