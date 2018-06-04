#!/bin/bash
	cd TestOneVSet
        pushd ..; make; popd;
	echo "if all goes well you'll see 10 runs of Training then the phrase **Test Complete**"
        rm -f onevset1.train onevset.test
        ../windows/svm-train -s 7 -t 0 -G 0.15 0.15 ./openset-example/leopards_training_0.05.data openset-example/leopards_training_0.05.data.model leo-m-7-0 >> onevset1.train;
	diff -w onevset1.train onevset1.train.out
        rm -f leo-m* junk
        rm -f onevset1.train
        echo "**Test Complete**"
        read -n 1 -s

