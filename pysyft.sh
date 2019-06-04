logDir="logs"

start=$(date +"%F_%T")
start_time=$(date +"%T")
echo "Starting execution: $start_time"

source ~/.bashrc
conda activate ./env

dstat --time -m --cpu --net --disk --output "$start".csv 60 > /dev/null
time python mainPySyft.py 2>&1 | tee "$start".txt

pkill dstat
end=$(date +"%T")
echo "Finished execution: $end"

if [ ! -d $logDir ]
then
    mkdir -p $logDir
fi

mv "$start".txt "$logDir/$start-$end".txt
mv "$start".csv "$logDir/$start-$end".csv
