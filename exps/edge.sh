python main.py --dataset=phone-email --ratio=2 --record
for edge_noise in $(seq 0.1 0.1 0.9); do
  python main.py --dataset=phone-email --ratio=2 --record --runs=10 --edge_noise=$edge_noise
done

python main.py --dataset=phone-email --ratio=2 --record
for edge_noise in $(seq 0.1 0.1 0.9); do
  python main.py --dataset=phone-email --ratio=2 --record --runs=10 --edge_noise=$edge_noise --robust
done

python main.py --dataset=Foursquare-Twitter --ratio=2 --record
for edge_noise in $(seq 0.1 0.1 0.9); do
  python main.py --dataset=Foursquare-Twitter --ratio=2 --record --runs=10 --edge_noise=$edge_noise
done

python main.py --dataset=Foursquare-Twitter --ratio=2 --record
for edge_noise in $(seq 0.1 0.1 0.9); do
  python main.py --dataset=Foursquare-Twitter --ratio=2 --record --runs=10 --edge_noise=$edge_noise --robust
done
