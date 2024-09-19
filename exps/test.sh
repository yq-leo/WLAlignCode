for edge_noise in $(seq 0.1 0.1 0.9); do
  echo "python main.py --dataset=phone-email --ratio=2 --record --runs=10 --edge_noise=$edge_noise"
done