#!/usr/bin/env python3
"""
Very simple HTTP server in python for logging requests
Usage::
    ./server.py [<port>]
"""
from http.server import BaseHTTPRequestHandler, HTTPServer
import logging, os, re
from datetime import datetime as dt
from urllib.parse import urlparse, parse_qs

class S(BaseHTTPRequestHandler):
    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        #print(re.sub('\?.*$', '', str(self.path)))
        if re.sub('\?.*$', '', str(self.path)) == "/refresh" or re.sub('\?.*$', '', str(self.path)) == "/export":
            os.system("/home/pi/waze/export_to_html.sh")
            self.wfile.write("Done".encode('utf-8'))
        elif re.sub('\?.*$', '', str(self.path)) == "/help":
            self.wfile.write("https://maps.googleapis.com/maps/api/js/AuthenticationService.Authenticate*".encode('utf-8'))
        elif re.sub('\?.*$', '', str(self.path)) == "/":
            #print(parse_qs(urlparse(self.path).query)["min"][0])
            od=0
            do=0

            t_od=0
            t_do=0
            try:
                od=int(dt.strptime(str(parse_qs(urlparse(self.path).query)["from"][0]), '%Y-%m-%dT%H:%M').timestamp() * 1000)
            except:
                pass
            try:
                do=int(dt.strptime(str(parse_qs(urlparse(self.path).query)["to"][0]), '%Y-%m-%dT%H:%M').timestamp() * 1000)
            except:
                pass


            cas=False
            try:
                t_od=dt.strptime(str(parse_qs(urlparse(self.path).query)["f_time"][0]), '%H:%M').time()
                cas=True
            except:
                pass
            try:
                t_do=dt.strptime(str(parse_qs(urlparse(self.path).query)["t_time"][0]), '%H:%M').time()
            except:
                cas=False
                pass

            if cas:
                if t_od == t_do:
                    cas = False

            with open('/home/pi/waze/files/start.html', 'rb') as file:
                self.wfile.write(file.read())
            with open('/home/pi/waze/database') as fp:
                for line in fp:
                    ob = line.strip().split(',')
                    boolL=True
                    if od > 0 and od > int(ob[0]):
                        boolL=False
                    if do > 0 and do < int(ob[0]):
                        boolL=False

                    if cas and boolL:
                        cteni = dt.fromtimestamp(int(ob[0])/1000).time()
                        # 21   0    02
                        # od   c    do
                        #192.168.0.137/?f_time=21%3A00&t_time=02%3A00
                        if t_od > t_do:
                            if cteni < t_do or cteni > t_od:
                                boolL=True
                            else:
                                boolL=False
                        else:
                            if t_od < cteni < t_do:
                                boolL=True
                            else:
                                boolL=False

                    if boolL:
                        #print(dt.fromtimestamp(int(ob[0])/1000))
                        self.wfile.write(str("new google.maps.LatLng(" + ob[1] + "," + ob[2] + "),").encode('utf-8'))
            with open('/home/pi/waze/files/end.html', 'rb') as file:
                self.wfile.write(file.read())

        elif str(self.path) == "/old":
            with open('/home/pi/waze/generated/web.html', 'rb') as file:
                self.wfile.write(file.read())
        elif str(self.path) == "/popunder.js":
            with open('/home/pi/waze/generated/popunder.js', 'rb') as file:
                self.wfile.write(file.read())
        #logging.info("GET request,\nPath: %s\nHeaders:\n%s\n", str(self.path), str(self.headers))
        #self._set_response()
        #self.wfile.write("GET request for {}".format(self.path).encode('utf-8'))

    def do_POST(self):
       # content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
       # post_data = self.rfile.read(content_length) # <--- Gets the data itself
       # logging.info("POST request,\nPath: %s\nHeaders:\n%s\n\nBody:\n%s\n",
        #        str(self.path), str(self.headers), post_data.decode('utf-8'))

        #self._set_response()
        self.wfile.write("POST request for {}".format(self.path).encode('utf-8'))

def run(server_class=HTTPServer, handler_class=S, port=80):
    #logging.basicConfig(level=logging.INFO)
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    #logging.info('Starting httpd...\n')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    logging.info('Stopping httpd...\n')

if __name__ == '__main__':
    from sys import argv

    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()
