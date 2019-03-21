#!/bin/bash

echo "test_empty_network"
expect="0"
got=`../a.out empty_network.txt 1 8`
diff <( echo "$expect" ) <( echo "$got" )

echo "test_singleton"
expect="0"
got=`../a.out singleton.txt 1 8`
diff <( echo "$expect" ) <( echo "$got" )

echo "test_singletons"
expect="0"
got=`../a.out singletons.txt 1 8`
diff <( echo "$expect" ) <( echo "$got" )

echo "test_single_edge"
expect="0.5"
got=`../a.out single_edge.txt 1 8`
diff <( echo "$expect" ) <( echo "$got" )

echo "test_disconnected_triads"
expect="0.666667"
got=`../a.out disconnected_triads.txt 1 8`
diff <( echo "$expect" ) <( echo "$got" )

echo "test_three_components_1"
expect="0.631579"
got=`../a.out three_components_1.txt 1 8`
diff <( echo "$expect" ) <( echo "$got" )

echo "test_three_components_2"
expect="0.636364"
got=`../a.out three_components_2.txt 1 8`
diff <( echo "$expect" ) <( echo "$got" )

echo "test_three_components_3"
expect="0.571429"
got=`../a.out three_components_3.txt 1 8`
diff <( echo "$expect" ) <( echo "$got" )

echo "test_zkc"
expect="2.33737"
got=`../a.out zkc.txt 1 8`
diff <( echo "$expect" ) <( echo "$got" )

