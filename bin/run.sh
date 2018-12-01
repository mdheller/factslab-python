#!/bin/bash
python run_baseline.py --prot arg --hand --search
sleep 2
python run_baseline.py --prot arg --hand --type --search
sleep 2
python run_baseline.py --prot arg --hand --token --search
sleep 2
python run_baseline.py --prot arg --hand --abl 1 --search
sleep 2
python run_baseline.py --prot arg --hand --abl 2 --search
sleep 2
python run_baseline.py --prot arg --hand --abl 3 --search
sleep 2
python run_baseline.py --prot arg --hand --abl 4 --search
sleep 2
python run_baseline.py --prot arg --hand --abl 5 --search
sleep 2
python run_baseline.py --prot arg --hand --abl 6 --search
sleep 2
python run_baseline.py --prot arg --hand --abl 7 --search
sleep 2
python run_baseline.py --prot arg --glove --search
sleep 2
python run_baseline.py --prot arg --elmo --search
sleep 2
python run_baseline.py --prot pred --hand --search
sleep 2
python run_baseline.py --prot pred --hand --token --search
sleep 2
python run_baseline.py --prot pred --hand --type --search
sleep 2
python run_baseline.py --prot pred --hand --abl 1 --search
sleep 2
python run_baseline.py --prot pred --hand --abl 2 --search
sleep 2
python run_baseline.py --prot pred --hand --abl 3 --search
sleep 2
python run_baseline.py --prot pred --hand --abl 4 --search
sleep 2
python run_baseline.py --prot pred --hand --abl 5 --search
sleep 2
python run_baseline.py --prot pred --hand --abl 6 --search
sleep 2
python run_baseline.py --prot pred --hand --abl 7 --search
sleep 2
python run_baseline.py --prot pred --glove --search
sleep 2
python run_baseline.py --prot pred --elmo --search
sleep 2
