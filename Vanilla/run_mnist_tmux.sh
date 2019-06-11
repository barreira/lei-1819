tmux new -d -s vanilla_mnist
tmux send-keys -t vanilla_mnist "./run_mnist.sh" "C-m"
