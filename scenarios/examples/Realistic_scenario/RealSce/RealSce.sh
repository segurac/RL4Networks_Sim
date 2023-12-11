#!/bin/bash
cp scratch/RealSce/Real_model-attributes.txt .
cp scratch/RealSce/satt_6c_r_mob_5000_ped_veh_r2.tcl ./scratch/
CXXFLAGS="-O3" ./waf configure -d debug --enable-examples --enable-tests
./waf build


for i in {1..250}
do
	./waf --run "scratch/RealSce/RealSce --RunNum=$(($i))"
done
