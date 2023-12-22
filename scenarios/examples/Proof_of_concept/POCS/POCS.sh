#! /bin/bash
cp scratch/POCS/LTE_Attributes.txt .

./ns3 configure --build-profile=optimized --enable-examples --enable-tests  --enable-mpi --enable-python
./ns3 build

for i in {1..250}
do

	./ns3 run "scratch/POCS/LTE_Environment --RunNum=$(($i))"
done
