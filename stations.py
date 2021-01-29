
from math import cos, asin, sqrt
import csv
import re
import geojson
import geobuf


class Station:

    def __str__(self):
        return f"{self.stn_name} ({self.stn_wmoid})"

    @classmethod
    def initialize_stations(cls):
        US_STATES = ["AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "IA", "ID",
                     "IL", "IN", "KS","LA", "MA", "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC",
                     "ND", "NE", "NH", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC", "SD",
                     "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY"]

        stations = geojson.FeatureCollection([])

        with open('station_list.txt', 'r') as csvfile:
            print("Running station initializer...")
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

                #print(stn_wmoid, stn_lat, stn_lon, stn_name, stn_altitude)

                if stn_altitude != -998.8:
                    point = geojson.Point((stn_lon, stn_lat, stn_altitude))
                    properties = {
                        "wmo_id": stn_wmoid,
                        "name":  stn_name
                    }
                    feature = geojson.Feature(geometry=point,
                                  properties=properties)
                    stations.features.append(feature)
            with open('station_list.pbf', 'wb') as gbfile:
                gb = geobuf.encode(stations)
                gbfile.write(gb)
            with open('station_list.geojson', 'wb') as gjfile:
                gj = geojson.dumps(stations, indent=4)
                gjfile.write(gj.encode("utf8"))
            print(len(gb),len(gj))



if __name__ == '__main__':

    Station.initialize_stations()
