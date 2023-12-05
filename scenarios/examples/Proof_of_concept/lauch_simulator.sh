#!/bin/bash

CURRENT_DIR=$(pwd)
SIM_NAME="POCS"


docker run --rm -it --net=host  -v ${CURRENT_DIR}/${SIM_NAME}/:/usr/ns-allinone-3.30/ns-3.30-mobility/scratch/${SIM_NAME}   ns3.30-mobility /bin/bash -c "cd /usr/ns-allinone-3.30/ns-3.30-mobility; ./scratch/${SIM_NAME}/${SIM_NAME}.sh"
