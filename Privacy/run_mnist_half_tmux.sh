tmux new -d -s privacy_mnist_half
tmux send-keys -t privacy_mnist_half "./run_mnist_half.sh" "C-m"
