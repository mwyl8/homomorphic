echo "Running FHE..."
nohup python3 fheesc50.py > fheesc50 2>&1
wait

echo "Running AHE EDB..."
nohup python3 aheedbesc50.py > aheedbesc50 2>&1
wait

echo "Running AHE EQV..."
nohup python3 aheeqvesc50.py > aheqvesc50 2>&1
wait

echo "All jobs finished."

