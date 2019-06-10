tmux new -d -s privacy_mnist
tmux send-keys -t privacy_mnist "./run_mnist.sh" "C-m"
