set -e

nohup python3 aheedb_fsdd_1024.py > aheedb_fsdd_1024 2>&1 &
pid1=$!
wait $pid1

nohup python3 aheedb_fsdd_128.py > aheedb_fsdd_128 2>&1 &
pid2=$!
wait $pid2

nohup python3 aheedb_fsdd_256.py > aheedb_fsdd_256 2>&1 &
pid3=$!
wait $pid3

nohup python3 aheedb_fsdd_512.py > aheedb_fsdd_512 2>&1 &
pid4=$!
wait $pid4


nohup python3 fhe_fsdd_1024.py > fhe_fsdd_1024 2>&1 &
pid5=$!
wait $pid5

nohup python3 fhe_fsdd_128.py > fhe_fsdd_128 2>&1 &
pid6=$!
wait $pid6

nohup python3 fhe_fsdd_256.py > fhe_fsdd_256 2>&1 &
pid7=$!
wait $pid7

nohup python3 fhe_fsdd_512.py > fhe_fsdd_512 2>&1 &
pid8=$!
wait $pid8


nohup python3 basetest_fsdd_1024.py > basetest_fsdd_1024 2>&1 &
pid9=$!
wait $pid9

nohup python3 basetest_fsdd_128.py > basetest_fsdd_128 2>&1 &
pid10=$!
wait $pid10

nohup python3 basetest_fsdd_256.py > basetest_fsdd_256 2>&1 &
pid11=$!
wait $pid11

nohup python3 basetest_fsdd_512.py > basetest_fsdd_512 2>&1 &
pid12=$!
wait $pid12


echo "All done"
