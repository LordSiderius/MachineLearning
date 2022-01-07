#!/bin/bash
fname="generated/"
fname+=$(date +"%Y_%m_%d-%H_%M")
fname+=".html"
echo "Creating file..."
cat files/start.html > $fname
echo "Generating coords..."
cat database | awk '{split($0,a,","); print "new google.maps.LatLng(" a[2] "," a[3] "),"}' >> $fname
cat files/end.html >> $fname
echo "Done!"

