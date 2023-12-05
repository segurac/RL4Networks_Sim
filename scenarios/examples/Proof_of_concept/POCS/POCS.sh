#! /bin/bash
cp scratch/POCS/LTE_Attributes.txt .
CXXFLAGS="-O3" ./waf configure -d debug --enable-examples --enable-tests
./waf build

for i in {1..250}
do
	./waf --run "scratch/POCS/POCS --RunNum=$(($i))"
done
