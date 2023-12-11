#!/bin/bash
CXXFLAGS="-O3" ./waf configure -d debug --enable-examples --enable-tests
./waf build
cp scratch/Power_CIO/satt_6c_r_mob_5000_ped_veh_r2.tcl ./scratch/

for ((i=1;i<=10000;i++))
do
        ./waf --run "scratch/Power_CIO/Power_CIO"
done


