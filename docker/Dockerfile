FROM ubuntu:20.04


ENV TZ=Europe/Madrid
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && \
    apt-get install -y \
        git \
        mercurial \
        gcc \
        g++ \
        vim \
        python \
        python-dev \
        python-setuptools \
        qt5-default \
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
        libboost-signals1.67-dev \
        libboost-filesystem1.67-dev \
        python3-dev \
        python3-virtualenv \
        python3.9-dev \
        python3.9-venv \
        python3.8-venv \
        build-essential

RUN mkdir -p /usr/ns3
WORKDIR /usr 

# RUN wget https://www.nsnam.org/release/ns-allinone-3.30.tar.bz2  && \
    # tar -jxvf ns-allinone-3.30.tar.bz2
RUN git clone https://github.com/segurac/ns-3-allinone ns-allinone-3.30

RUN cd ns-allinone-3.30 && ./download.py -n ns-3.30-mobility

RUN cd ns-allinone-3.30/ns-3.30-mobility/ && CXXFLAGS="-O3" ./waf --enable-tests --enable-examples configure

RUN cd ns-allinone-3.30/ns-3.30-mobility/ && ./waf build

# RUN cd ns-allinone-3.30 && ./build.py --enable-examples --enable-tests

RUN ln -s /usr/ns-allinone-3.30/ns-3.30-mobility/ /usr/ns3/

RUN apt-get clean

##### ns3gym

RUN apt-get install --yes libzmq5 libzmq3-dev libprotobuf-dev protobuf-compiler

RUN cd ns-allinone-3.30/ns-3.30-mobility/contrib && \
    git clone https://github.com/tkn-tub/ns3-gym.git ./opengym && \
    cd opengym/ && \
    git checkout app

##Compile again NS3
RUN cd ns-allinone-3.30/ns-3.30-mobility/ && \
    ./waf configure --enable-examples && \
    ./waf build

RUN apt-get install -y python3-pip

RUN cd ns-allinone-3.30/ns-3.30-mobility/contrib/opengym/ && \
    pip3 install --user ./model/ns3gym

## Test that it works
#cd ns-allinone-3.30/ns-3.30-mobility/contrib/opengym/examples/opengym/ 
#./simple_test.py
# esto para otra versión más antigua de protocol buffers
# PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python ./simple_test.py



