#!/bin/bash
python run_baseline.py --prot pred --hand --abl 1
sleep 5
python run_baseline.py --prot pred --hand --abl 2
sleep 5
python run_baseline.py --prot pred --elmo --abl 3
sleep 5
python run_baseline.py --prot pred --elmo --abl 4
sleep 5
python run_baseline.py --prot pred --elmo --abl 5
sleep 5
python run_baseline.py --prot pred --elmo --abl 6
sleep 5
python run_baseline.py --prot pred --elmo --abl 7
