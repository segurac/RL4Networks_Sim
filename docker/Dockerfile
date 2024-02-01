FROM ubuntu:22.04


ENV TZ=Europe/Madrid
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


RUN apt-get update

RUN DEBIAN_FRONTEND=noninteractive apt install --yes \
        git \
        mercurial \
        gcc \
        g++ \
        vim \
        autoconf \
        cvs \
        bzr \
        unrar \
        gdb \
        valgrind \
        uncrustify \
        flex \
        bison \
        libfl-dev \
        tcpdump \
        gsl-bin \
        libgsl-dev \
        sqlite \
        sqlite3 \
        libsqlite3-dev \
        libxml2 \
        libxml2-dev \
        cmake \
        libc6-dev \
        libc6-dev-i386 \
        libclang-dev \
        llvm-dev \
        automake \
        libgtk2.0-0 \
        libgtk2.0-dev \
        vtun \
        lxc \
        libboost-dev \
        python3-dev \
        python3-virtualenv \
        python3.10-dev \
        python3.10-venv \
        python3.11-venv \
        python3.11-dev \
        build-essential \
        libzmq5 \
        libzmq3-dev \
        libprotobuf-dev \
        protobuf-compiler \
        pkg-config \
        python3-setuptools \
        gir1.2-gtk-3.0  gir1.2-goocanvas-2.0 \
        python3-gi python3-gi-cairo python3-pygraphviz \
        openmpi-bin openmpi-common openmpi-doc libopenmpi-dev \
        autoconf \
        unrar \
        tcpdump wireshark \
        libxml2 libxml2-dev \
        qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools \
        libgcrypt20-dev \
        libgtk-3-dev \
        python3-pygccxml \
        ccache \ 
        \cmake-format \
        libeigen3-dev && \
        apt-get clean


RUN mkdir -p /usr/ns3
WORKDIR /usr 

ADD https://api.github.com/repos/segurac/ns-3-dev-git/git/refs/heads/dev-mobility version.json

RUN git clone https://github.com/segurac/ns-3-allinone ns-allinone-3.30

RUN cd ns-allinone-3.30 && ./download.py -n dev-mobility && mv dev-mobility ns-3-mobility

RUN python3 -m pip install --user cppyy==2.4.2

RUN cd ns-allinone-3.30/ns-3-mobility/ && \
    ./ns3 configure --build-profile=optimized --enable-examples --enable-tests  --enable-mpi --enable-python -- -DNS3_BINDINGS_INSTALL_DIR="/root/.local/lib/python3.10/site-packages"


RUN cd ns-allinone-3.30/ns-3-mobility/ && ./ns3 build

ADD https://api.github.com/repos/tkn-tub/ns3-gym/git/refs/heads/app-ns-3.36+ version_gym.json

RUN cd ns-allinone-3.30/ns-3-mobility/contrib && \
    git clone https://github.com/tkn-tub/ns3-gym.git ./opengym && \
    cd opengym/ && \
    git checkout app-ns-3.36+

# Patch it
COPY ns3env_numpy.patch /tmp/
RUN patch ns-allinone-3.30/ns-3-mobility/contrib/opengym/model/ns3gym/ns3gym/ns3env.py < /tmp/ns3env_numpy.patch


##Compile again NS3
RUN cd ns-allinone-3.30/ns-3-mobility/ && \
    ./ns3 configure --build-profile=optimized --enable-examples --enable-tests  --enable-mpi --enable-python && \
    ./ns3 build

RUN cd ns-allinone-3.30/ns-3-mobility/contrib/opengym/ && \
    pip3 install --user ./model/ns3gym && \
    pip install --user protobuf

## Generate a .whl file for the client
RUN cd ns-allinone-3.30/ns-3-mobility/contrib/opengym/model/ns3gym && \
    python3 setup.py bdist_wheel


#ADD https://api.github.com/repos/segurac/ns-3-dev-git/git/refs/heads/dev-mobility version.json


RUN cd ns-allinone-3.30/ns-3-mobility/ && \
    git pull  && \
    ./ns3 configure --build-profile=optimized --enable-examples --enable-tests  --enable-mpi --enable-python -- -DNS3_BINDINGS_INSTALL_DIR="/root/.local/lib/python3.10/site-packages" && \
    ./ns3 build
