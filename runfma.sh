set -e

nohup python3 aheedbfma.py > aheedbfma 2>&1 &
pid1=$!
wait $pid1

nohup python3 fhefma.py > fhefma 2>&1 &
pid2=$!
wait $pid2

nohup python3 aheeqvfma.py > aheeqvfma 2>&1 &

echo "All done"
