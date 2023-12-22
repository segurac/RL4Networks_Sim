#!/bin/bash
cp scratch/RealSce/Real_model-attributes.txt .
cp scratch/RealSce/satt_6c_r_mob_5000_ped_veh_r2.tcl ./scratch/

./ns3 configure --build-profile=optimized --enable-examples --enable-tests  --enable-mpi --enable-python
./ns3 build


for i in {1..250}
do
	./ns3 run "scratch/RealSce/LTE_Environment --RunNum=$(($i))"
done
