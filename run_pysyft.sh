set_epochs() {
	sed -i -e "s/\(self\.epochs = \).*$/\1$1/g" "./mainPySyft.py"
}

set_batch_size() {
	sed -i -e "s/\(self\.batch_size = \).*$/\1$1/g" "./mainPySyft.py"
}

set_epochs 10
set_batch_size 256

tmux new -d -s execute
tmux send-keys -t execute "./pysyft.sh" "C-m"
