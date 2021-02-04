
import os
import sys
import csv
import re
import geojson
import json

def newer(filename, ext2):
    (fn, ext) = os.path.splitext(os.path.basename(filename))
    target = fn + ext2
    if not os.path.exists(target):
        return True
    return os.path.getmtime(filename) > os.path.getmtime(target)


def initialize_stations(filename):
    US_STATES = ["AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "IA", "ID",
                 "IL", "IN", "KS","LA", "MA", "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC",
                 "ND", "NE", "NH", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC", "SD",
                 "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY"]

    stations = geojson.FeatureCollection([])
    stationdict = dict()
    with open(filename, 'r') as csvfile:
        stndata = csv.reader(csvfile, delimiter='\t')
        for row in stndata:
            m = re.match(r"(?P<stn_wmoid>^\w+)\s+(?P<stn_lat>\S+)\s+(?P<stn_lon>\S+)\s+(?P<stn_altitude>\S+)(?P<stn_name>\D+)" , row[0])
            fields = m.groupdict()
            stn_wmoid = fields['stn_wmoid'][6:]
            stn_name = fields['stn_name'].strip()

            if re.match(r"^[a-zA-Z]{2}\s", stn_name) and  stn_name[:2] in US_STATES:
                stn_name = stn_name[2:].strip().title() + ", " + stn_name[:2]
            else:
                stn_name = stn_name.title()
            stn_name = fields['stn_name'].strip().title()
            stn_lat = float(fields['stn_lat'])
            stn_lon = float(fields['stn_lon'])
            stn_altitude = float(fields['stn_altitude'])

            if stn_altitude > -998.8:
                stationdict[stn_wmoid] = {
                    "name":  stn_name,
                    "lat": stn_lat,
                    "lon" : stn_lon,
                    "elevation": stn_altitude
                }

        (fn, ext) = os.path.splitext(os.path.basename(filename))
        with open(f'{fn}.json', 'wb') as jfile:
            j = json.dumps(stationdict, indent=4).encode("utf8")
            jfile.write(j)


if __name__ == '__main__':
    fn = "station_list.txt"
    if newer(fn, ".json"):
        print("time to rebuild")
        initialize_stations(fn)
