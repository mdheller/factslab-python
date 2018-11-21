#!/bin/bash
python run_baseline.py --prot arg --hand --embed
sleep 5
python run_baseline.py --prot pred --hand --embed
sleep 5
python run_baseline.py --prot arg --hand
sleep 5
python run_baseline.py --prot pred --hand
sleep 5
python run_baseline.py --prot arg --embed
sleep 5
python run_baseline.py --prot pred --embed
sleep 5