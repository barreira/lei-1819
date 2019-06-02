logDir="/home/admin/PySyft/logs"

start=$(date +"%F_%T")
start_time=$(date +"%T")
echo "Starting execution: $start_time"

source /home/admin/.bashrc
conda activate /home/admin/PySyft/env

dstat --time -m --cpu --net --output "$start".csv 1
time python mnist.py | tee "$start".txt

pkill dstat
end=$(date +"%T")
echo "Finished execution: $end"

if [ ! -d $logDir ]
then
    mkdir -p $logDir
fi

mv "$start".txt "$logDir/$start-$end".txt
mv "$start".csv "$logDir/$start-$end".csv
