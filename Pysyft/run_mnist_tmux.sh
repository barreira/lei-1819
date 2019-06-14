tmux new -d -s pysyft_mnist
tmux send-keys -t pysyft_mnist "./run_mnist.sh" "C-m"
