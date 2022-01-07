#!/bin/bash
lines=$(wc -l database |cut -d" " -f1)
echo "Downloading info..."
curl --silent 'https://www.waze.com/row-rtserver/web/TGeoRSS?bottom=49.8963338912359&left=13.978387430545535&ma=200&mj=100&right=14.880639627811156&top=50.26013173969895&types=alerts%2Ctraffic' > /tmp/waze_data
echo "Extracting POLICE alerts from JSON..."
jq -r '.alerts[]|select(.type=="POLICE")| "\(.pubMillis),\(.location.y),\(.location.x)"' /tmp/waze_data >> database
echo "Done!"
echo New lines: $(($(wc -l database |cut -d" " -f1)-$lines))
