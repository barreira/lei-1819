tmux new -d -s tfencrypted_mnist_half
tmux send-keys -t tfencrypted_mnist_half "./run_mnist_half.sh" "C-m"
