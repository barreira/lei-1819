tmux new -d -s tfencrypted_mnist
tmux send-keys -t tfencrypted_mnist "./run_mnist.sh" "C-m"
