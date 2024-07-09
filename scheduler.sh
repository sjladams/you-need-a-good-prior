echo "run [2x64] ls=1"
python notebooks/prior1d.py --con 0 --width 64 --depth 2 --ls 1.0

echo "run [2x128] ls=0.75"
python notebooks/prior1d.py --con 0 --width 128 --depth 2 --ls 0.75

echo "run [2x128] ls=1.0"
python notebooks/prior1d.py --con 0 --width 128 --depth 2 --ls 1.0