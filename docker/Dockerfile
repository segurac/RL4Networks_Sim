FROM ubuntu:22.04


ENV TZ=Europe/Madrid
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && \
    apt-get install -y \
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
        protobuf-compiler

RUN mkdir -p /usr/ns3
WORKDIR /usr 

# RUN wget https://www.nsnam.org/release/ns-allinone-3.30.tar.bz2  && \
    # tar -jxvf ns-allinone-3.30.tar.bz2
RUN git clone https://github.com/segurac/ns-3-allinone ns-allinone-3.30

RUN cd ns-allinone-3.30 && ./download.py -n ns-3.30-mobility

RUN cd ns-allinone-3.30/ns-3.30-mobility/ && git reset --hard c972449f774c268fd20b82e26245d6584978fe86

RUN cd ns-allinone-3.30/ns-3.30-mobility/ && CXXFLAGS="-O3" ./waf --enable-tests --enable-examples configure

RUN cd ns-allinone-3.30/ns-3.30-mobility/ && ./waf build

# RUN cd ns-allinone-3.30 && ./build.py --enable-examples --enable-tests

RUN ln -s /usr/ns-allinone-3.30/ns-3.30-mobility/ /usr/ns3/

RUN apt-get clean

##### ns3gym

RUN cd ns-allinone-3.30/ns-3.30-mobility/contrib && \
    git clone https://github.com/tkn-tub/ns3-gym.git ./opengym && \
    cd opengym/ && \
    git checkout app

#### Apply patch to ns3env to be compatible with new numpy

COPY ns3env_numpy.patch /tmp/
RUN patch ns-allinone-3.30/ns-3.30-mobility/contrib/opengym/model/ns3gym/ns3gym/ns3env.py < /tmp/ns3env_numpy.patch


##Compile again NS3
RUN cd ns-allinone-3.30/ns-3.30-mobility/ && \
    CXXFLAGS="-O3" ./waf configure --enable-examples && \
    ./waf build

RUN cd ns-allinone-3.30/ns-3.30-mobility/contrib/opengym/ && \
    pip3 install ./model/ns3gym && \
    pip install protobuf==3.20.3    


## Generate a .whl file for the client
RUN cd ns-allinone-3.30/ns-3.30-mobility/contrib/opengym/model/ns3gym && \
    python3 setup.py bdist_wheel


## Test that it works
#cd ns-allinone-3.30/ns-3.30-mobility/contrib/opengym/examples/opengym/ 
#./simple_test.py
# esto para otra versión más antigua de protocol buffers
# PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python ./simple_test.py

#Since this line changes for every commit, it will effectively rebuild everything after this
ADD https://api.github.com/repos/segurac/ns-3-dev-git/git/refs/heads/ns-3.30-mobility version.json


RUN cd ns-allinone-3.30/ns-3.30-mobility/ && \
    git pull  && \
    CXXFLAGS="-O3" ./waf configure -d debug --enable-examples --enable-tests && \
    ./waf




