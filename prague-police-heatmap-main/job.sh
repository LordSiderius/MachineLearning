while true
do
lines=$(wc -l database |cut -d" " -f1)
echo "Fetching..."
./fetch_waze.sh > /dev/null
cas=$(shuf -i 500-1200 -n 1)
date
echo New lines: $(($(wc -l database |cut -d" " -f1)-$lines))
echo "Sleeping for $cas seconds"
sleep $cas

done
