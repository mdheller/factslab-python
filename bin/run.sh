#!/bin/bash
python run_baseline.py --search --prot arg --hand --embed
sleep 5
python run_baseline.py --search --prot pred --hand --embed
sleep 5
python run_baseline.py --search --prot arg --hand
sleep 5
python run_baseline.py --search --prot pred --hand
sleep 5
python run_baseline.py --search --prot arg --embed
sleep 5
python run_baseline.py --search --prot pred --embed
sleep 5