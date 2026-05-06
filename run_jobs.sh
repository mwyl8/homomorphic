#!/bin/bash
python3 fhe6.py > fhe6results 2>&1
python3 aheeqv.py > aheeqvresults 2>&1
python3 aheedb.py > aheedbresults 2>&1
python3 base.py > baseresults 2>&1

