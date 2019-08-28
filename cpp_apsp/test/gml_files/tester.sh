#!/bin/bash
echo "Output:"
/Users/alexray/Dropbox/graph_scaling/empirical_network_scaling/cpp_apsp/a.out test_files_list.txt output.txt 8
echo "Results:"
python3 corroborator.py
rm output.txt
