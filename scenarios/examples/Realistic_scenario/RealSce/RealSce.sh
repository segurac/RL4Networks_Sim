#! /bin/bash
cp scratch/RealSce/Real_model-attributes.txt .
CXXFLAGS="-O3" ./waf configure -d debug --enable-examples --enable-tests
./waf build


for i in {1..250}
do
	./waf --run "scratch/RealSce/RealSce --RunNum=$(($i))"
done
