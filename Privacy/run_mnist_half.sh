set_epochs() {
	sed -i -e "s/epochs', [0-9]*/epochs', $1/g" "./mnist_half.py"
}

set_batch_size() {
	sed -i -e "s/batch_size', [0-9]*/batch_size', $1/g" "./mnist_half.py"
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
		log_dir="mnist-half-e$x-b$j"
		
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
		time python mnist_half.py 2>&1 | tee "$start".txt

		pkill dstat
		end=$(date +"%T")

		mv "$start".txt "$log_dir/$start-$end".txt
		mv "$start".csv "$log_dir/$start-$end".csv
	done
done
