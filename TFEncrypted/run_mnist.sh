set_epochs() {
	sed -i -e "s/EPOCHS = [0-9]*/EPOCHS = $1/g" "./mnist.py"
}

set_batch_size() {
	sed -i -e "s/BATCH_SIZE = [0-9]*/BATCH_SIZE = $1/g" "./mnist.py"
}

epochs=(1 5 10)
batch_sizes=(128 256 512 1024)
log_dir="dir"
x="1"

for i in "${epochs[@]}"
do
	set_epochs $i
	x="$i"
	for j in "${batch_sizes[@]}"
	do
		log_dir="logs/8gb/mnist_full/mnist-e$x-b$j"
		
		if [ ! -d $log_dir ]
		then
			mkdir -p $log_dir
		fi

		set_batch_size $j
		start=$(date +"%F_%T")
		start_time=$(date +"%T")
		
		echo "Epochs: $i Batch Size: $j"

		source ~/.bashrc
		conda activate ./env

		dstat --time -m --cpu --net --disk --output "$start".csv 60 > /dev/null &
		time python mnist.py 2>&1 | tee "$start".txt

		pkill dstat
		end=$(date +"%T")

		mv "$start".txt "$log_dir/$start-$end".txt
		mv "$start".csv "$log_dir/$start-$end".csv
	done
done
