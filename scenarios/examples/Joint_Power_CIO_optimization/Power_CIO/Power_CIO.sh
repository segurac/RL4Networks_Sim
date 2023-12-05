#!/bin/bash
CXXFLAGS="-O3" ./waf configure -d debug --enable-examples --enable-tests
./waf build


for ((i=1;i<=10000;i++))
do
        ./waf --run "scratch/Power_CIO/Power_CIO"
done


