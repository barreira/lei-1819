tmux new -d -s vanilla_mnist_half
tmux send-keys -t vanilla_mnist_half "./run_mnist_half.sh" "C-m"
