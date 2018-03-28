#!/bin/bash

python main.py --lr=0.1
python main.py --resume --lr=0.01
python main.py --resume --lr=0.001

