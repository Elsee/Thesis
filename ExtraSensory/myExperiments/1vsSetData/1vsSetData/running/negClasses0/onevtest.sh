#!/bin/bash
	cd TestOneVSet
        pushd ..; make; popd;
	echo "if all goes well you'll see 10 runs of Training then the phrase **Test Complete**"
        for u in 2 3 4 6 10 16 20 26 27 29 30 32 38 39 41 43 45 46 49 53 54 56 ; do  ../../libSVM/svm-train -s 7 -t 0 train#$u.data extra-m-0-7 >> onevset.train; ../../libSVM/svm-predict  test#$u.data extra-m-0-7 junk >> onevset.test#$u  ;done
	
        rm -f extra-m* junk
        echo "**Test Complete**"
        read -n 1 -s


