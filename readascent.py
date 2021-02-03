#  This program was automatically generated with bufr_dump -Dpython
#  Using ecCodes version: 2.19.1
#
# bufr_dump -Dpython A_IUSD02LOWM210300_C_EDZW_20210121040000_59339751.bin >dump.py
# and hand edited.

import traceback
import sys
from eccodes import *
import argparse
import orjson
import geojson
import logging
import datetime
import os
from math import atan, sin, cos
import brotli
import pathlib
import zipfile
import tempfile
import ciso8601
from pprint import pprint
from operator import itemgetter
from string import punctuation

earth_avg_radius =  6371008.7714
earth_gravity =  9.80665

# metpy is terminally slow, so roll our own sans dimension checking
def geopotential_height_to_height(gph):
    geopotential = gph * earth_gravity
    return (geopotential * earth_avg_radius) / (earth_gravity * earth_avg_radius - geopotential)


class OneLineExceptionFormatter(logging.Formatter):
    def formatException(self, exc_info):
        result = super(OneLineExceptionFormatter, self).formatException(exc_info)
        return repr(result) # or format into one line however you want to

    def format(self, record):
        s = super(OneLineExceptionFormatter, self).format(record)
        if record.exc_text:
            s = s.replace('\n', '') + '|'
        return s


def bufr_decode(f, args):

    ibufr = codes_bufr_new_from_file(f)
    if not ibufr:
        raise Exception("empty file")
    codes_set(ibufr, 'unpack', 1)

    missingHdrKeys = 0

    header = dict()
    arraykeys = ['delayedDescriptorReplicationFactor',
                 'extendedDelayedDescriptorReplicationFactor',
                 'unexpandedDescriptors']

    for k in arraykeys:
        try:
            header[k] = list(codes_get_array(ibufr, k))
        except Exception as e:
            logging.info(f"array key in header key={k} e={e}")
            missingHdrKeys += 1

    num_samples = header['extendedDelayedDescriptorReplicationFactor'][0]
    #logging.info(f"num_samples={num_samples}")

    scalarkeys = [
#        'edition',
#        'masterTableNumber',
#        'bufrHeaderCentre',
#        'bufrHeaderSubCentre',
#        'updateSequenceNumber',
#        'dataCategory',
#        'internationalDataSubCategory',
#        'dataSubCategory',
#        'masterTablesVersionNumber',
#        'localTablesVersionNumber',
        'typicalYear',
        'typicalMonth',
        'typicalDay',
        'typicalHour',
        'typicalMinute',
        'typicalSecond',
        'typicalDate',
        'typicalTime',
#        'numberOfSubsets',
#        'observedData',
#        'compressedData',
        'radiosondeSerialNumber',
#        'radiosondeAscensionNumber',
#        'radiosondeReleaseNumber',
#        'observerIdentification',
#        'radiosondeCompleteness',
#        'radiosondeConfiguration',
#        'correctionAlgorithmsForHumidityMeasurements',
#        'radiosondeGroundReceivingSystem',
        'radiosondeOperatingFrequency',
#        'balloonManufacturer',
#        'weightOfBalloon',
#        'amountOfGasUsedInBalloon',
#        'pressureSensorType',
#        'temperatureSensorType',
#        'humiditySensorType',
#        'geopotentialHeightCalculation',
#        'softwareVersionNumber',
#        'reasonForTermination',
        'blockNumber',
        'stationNumber',
        'radiosondeType',
#        'solarAndInfraredRadiationCorrection',
#        'trackingTechniqueOrStatusOfSystem',
#        'measuringEquipmentType',
#        'timeSignificance',
        'year',
        'month',
        'day',
        'hour',
        'minute',
        'second',
        'latitude',
        'longitude',
        'heightOfStationGroundAboveMeanSeaLevel',
        'heightOfBarometerAboveMeanSeaLevel',
        'height',
#        'shipOrMobileLandStationIdentifier',
        #'text'
        ]

    for k in scalarkeys:
        try:
            header[k] = codes_get(ibufr, k)
        except Exception as e:
            logging.debug(f"scalar hdr key={k} e={e}")
            missingHdrKeys += 1

    # special-case warts we do not really care about
    warts = ['shipOrMobileLandStationIdentifier'

            ]

    for k in warts:
        try:
            header[k] = codes_get(ibufr, k)
        except Exception:
            missingHdrKeys += 1

    keys = ['timePeriod',
            # 'extendedVerticalSoundingSignificance',
            'pressure', 'nonCoordinateGeopotentialHeight',
            'latitudeDisplacement', 'longitudeDisplacement',
            'airTemperature', 'dewpointTemperature',
            'windDirection', 'windSpeed']

    samples = []
    invalidSamples = 0
    missingValues = 0
    for i in range(1, num_samples + 1):
        sample = dict()
        for k in keys:
            name = f"#{i}#{k}"
            try:
                sample[k] = codes_get(ibufr, name)
            except Exception as e:
                logging.debug(f"sample={i} key={k} e={e}, skipping")
                missingValues += 1
        # call BS on bogus values
        if float(sample['airTemperature']) < -273 or float(sample['dewpointTemperature']) < -273:
            invalidSamples += 1
            continue
        samples.append(sample)
    logging.debug((f"samples used={len(samples)}, invalid samples="
                   f"{invalidSamples}, skipped header keys={missingHdrKeys},"
                   f" missing values={missingValues}"))

    codes_release(ibufr)
    return header, samples


rad = 4.0 * atan(1) / 180.


def winds_to_UV(windSpeed, windDirection):
    u = -windSpeed * sin(rad * windDirection)
    v = -windSpeed * cos(rad * windDirection)
    return u, v


def gen_id(h):
    bn = h.get('blockNumber', 2147483647)
    sn = h.get('stationNumber', 2147483647)

    if (bn != 2147483647 and sn != 2147483647):
        return ("wmo", f'{bn:02d}{sn:03d}')

    if 'shipOrMobileLandStationIdentifier' in h:
        id = h['shipOrMobileLandStationIdentifier']
        # if it looks remotely like an id...
        if not any(p in id for p in punctuation):
            return ("mobile", h['shipOrMobileLandStationIdentifier'])

    return ("location", f"{h['latitude']:.3f}:{h['longitude']:.3f}")

def convert_to_geojson(args, h, samples):
    takeoff = datetime.datetime(year=h['year'],
                                month=h['month'],
                                day=h['day'],
                                minute=h['minute'],
                                hour=h['hour'],
                                second=h['second'],
                                tzinfo=None)

    typ, id = gen_id(h)

    ts = ciso8601.parse_datetime(h['typicalDate'] + " " +  h['typicalTime']).timestamp()

    properties = {
        "station_id":  id,
        "id_type":  typ,
        "syn_date": h['typicalDate'],
        "syn_time": h['typicalTime'],
        "syn_timestamp": ts,
        "firstSeen": takeoff.timestamp(),
        "lat": h['latitude'],
        "lon": h['longitude'],
        "station_elevation": h['heightOfStationGroundAboveMeanSeaLevel'],  # ?
        "baro_elevation": h['heightOfBarometerAboveMeanSeaLevel'],  # ?
#        "sonde_serial": h['radiosondeSerialNumber'],
#        "sonde_frequency":  h['radiosondeOperatingFrequency'],
        "sonde_type":  h['radiosondeType']
    }
    # patch up missing ascent elevation with station elevation
    if 'height' in h:
        properties['elevation'] = h['height']
    else:
        properties['elevation'] = h['heightOfStationGroundAboveMeanSeaLevel']

    if 'radiosondeSerialNumber' in h:
        properties['sonde_serial'] = h['radiosondeSerialNumber']

    if 'radiosondeOperatingFrequency' in h:
        properties['sonde_frequency'] = h['radiosondeOperatingFrequency']

    fc = geojson.FeatureCollection([])
    fc.properties = properties
    lat_t = fc.properties['lat']
    lon_t = fc.properties['lon']
    firstSeen = fc.properties['firstSeen']
    previous_elevation = fc.properties['elevation'] - args.hstep

    for s in samples:
        lat = lat_t + s['latitudeDisplacement']
        lon = lon_t + s['longitudeDisplacement']
        gpheight = s['nonCoordinateGeopotentialHeight']

        delta = datetime.timedelta(seconds=s['timePeriod'])
        sampleTime = takeoff + delta

        height = geopotential_height_to_height(gpheight)
        if height < previous_elevation + args.hstep:
            continue
        previous_elevation = height

        u, v = winds_to_UV(s['windSpeed'], s['windDirection'])
        if args.winds_dir_speed:
            winds = {
                "wdir": s['windDirection'],
                "wspeed": s['windSpeed']
            }
        else:
            winds = {
                "wind_u": u,
                "wind_v": v
            }

        properties = {
            "time": sampleTime.timestamp(),
            "gpheight": gpheight,
            "temp": s['airTemperature'],
            "dewpoint": s['dewpointTemperature'],
            "pressure": s['pressure'],
        }
        f = geojson.Feature(geometry=geojson.Point((lon, lat, height)),
                            properties={**properties, **winds})
        fc.features.append(f)
    fc.properties['lastSeen'] = sampleTime.timestamp()
    return fc


def gen_czml(fc):
    logging.warning(f"not implemented yet")


updated_stations = []

def gen_output(args, h, s, fn, zip):
    h['samples'] = s
    if args.orig:
        print(h)

    if len(s) < 10:
        logging.info(f'QC: skipping {fn} from {zip} - only {len(s)} samples')
        return

    # QC here!
    if not ({'year', 'month', 'day', 'minute', 'hour', 'second'} <= h.keys()):
        logging.info(f'QC: skipping {fn} from {zip}')
        return
    if h['year'] == 2147483647:
        logging.info(f'QC: skipping {fn} from {zip}, no date')
        return

    if h['height'] == 2147483647:
        # delete BS key, will be filled in by station elevation
        del h['height']

    if  h['second'] > 59:
        h['second'] = 0


    fc = convert_to_geojson(args, h, s)
    fc.properties['origin_fn'] = fn
    if zip:
        fc.properties['origin_zip'] = pathlib.PurePath(zip).name
    station_id = fc.properties['station_id']
    logging.debug(
        f'output samples retained: {len(fc.features)}, station id={station_id}')

    updated_stations.append((station_id, fc.properties))

    cext = ""
    if args.brotli:
        cext = ".br"

    cc = station_id[:2]
    day = fc.properties['syn_date']
    time = fc.properties['syn_time']

    if args.by_basename:
        dest = f'{args.destdir}/{fn}.geojson{cext}'
        ref = f'{fn}.geojson'
    else:
        dest = f'{args.destdir}/{cc}/{station_id}_{day}_{time}.geojson{cext}'
        ref = f'{cc}/{station_id}_{day}_{time}.geojson'

    if args.mkdirs:
        path = pathlib.Path(dest).parent.absolute()
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    gj = geojson.dumps(fc).encode("utf8")
    if args.geojson:
        logging.debug(f'writing {dest}')
        with open(dest, 'wb') as gjfile:
            cmp = gj
            if args.brotli:
                cmp = brotli.compress(gj)
            gjfile.write(cmp)

    fc.properties['path'] = ref
    if args.dump_geojson:
        print(gj)


def process(args, f, fn, zip):
    try:
        (h, s) = bufr_decode(f, args)

    except Exception as e:
        logging.info(f"exception processing {fn} from {zip}: {e}")

    except CodesInternalError as err:
        traceback.print_exc(file=sys.stderr)
    else:
        gen_output(args, h, s, fn, zip)

def patchup(ascents):
    # remove any cruft which crept in
    for a in ascents:
        e = a['elevation']
        if e == 2147483647:
            a['elevation'] = a['station_elevation']
            logging.info(f"---- patchup {a['station_id']} elevation={a['elevation']}")
        if 'sonde_type' in a and a['sonde_type'] == 2147483647:
            del a['sonde_type']
            logging.info(f"---- patchup {a['station_id']} delete sonde_type")

def update_summary(args, updated_stations):
    if args.summary and os.path.exists(args.summary):
        with open(args.summary, 'rb') as f:
            s = f.read().decode()
            summary = geojson.loads(s)
            logging.debug(f'read summary from {args.summary}')
    else:
        logging.info(f'no summary file')
        summary = dict()

    if args.stations and os.path.exists(args.stations):
        with open(args.stations, 'rb') as f:
            s = f.read().decode()
            stations = orjson.loads(s)
            logging.debug(f'read stations from {args.stations}')
    else:
        logging.info(f'no stations file')
        stations = dict()

    for id, desc in summary.items():
        patchup(desc['ascents'])

    for id, asc in  updated_stations:
        if id in summary:
            # append, sort and de-duplicate
            oldlist = summary[id]['ascents']
            #patchup(asc)
            oldlist.append(asc)
            newlist = sorted(oldlist, key=itemgetter('syn_timestamp'), reverse=True)
            # https://stackoverflow.com/questions/9427163/remove-duplicate-dict-in-list-in-python
            seen = set()
            dedup = []
            for d in newlist:
                t = d['syn_timestamp']
                if t not in seen:
                    seen.add(t)
                    dedup.append(d)
            summary[id]['ascents'] = dedup
        else:
            if id in stations:
                st =  stations[id]
            else:
                # anonymous + mobile stations
                st = {
                    "name": asc['station_id'],
                    "lat" : asc['lat'],
                    "lon" : asc['lon'],
                    "elevation" : asc['elevation']
                }
            st['ascents'] = [asc]
            summary[id] = st

        js = orjson.dumps(summary, option=orjson.OPT_INDENT_2)
        fd, path = tempfile.mkstemp(dir=args.tmpdir)
        os.write(fd, js)
        os.fsync(fd)
        os.close(fd)
        os.rename(path, args.summary)
        os.chmod(args.summary, 0o644)

        # create GeoJSON version of summary
        gj = json2geojson(summary)
        dest = os.path.splitext(args.summary)[0]+'.geojson'
        fd, path = tempfile.mkstemp(dir=args.tmpdir)
        os.write(fd, gj.encode("utf8"))
        os.fsync(fd)
        os.close(fd)
        os.rename(path, dest)
        os.chmod(dest, 0o644)

def json2geojson(sj):
    fc = geojson.FeatureCollection([])

    for id, desc in sj.items():
        pt = (desc['lon'], desc['lat'], desc['elevation'])
        deleatur = ['lon', 'lat', 'elevation']
        newdict = {k: v for k, v in desc.items() if not k in deleatur}

        f = geojson.Feature(geometry=geojson.Point(pt),
                            properties=newdict)
        fc.features.append(f)
    return geojson.dumps(fc, indent=4)

def main():
    parser = argparse.ArgumentParser(description='decode radiosonde BUFR report',
                                     add_help=True)
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parser.add_argument('--orig', action='store_true', default=False)
    parser.add_argument('--hstep', action='store', type=int, default=0)
    parser.add_argument('--winds-dir-speed',
                        action='store_true', default=False)
    parser.add_argument('--destdir', action='store', default=".")
    parser.add_argument('--geojson', action='store_true', default=False)
    parser.add_argument('--dump-geojson', action='store_true', default=False)
    parser.add_argument('--brotli', action='store_true', default=False)
    parser.add_argument('--by-basename', action='store_true', default=False)
    parser.add_argument('--mkdirs', action='store_true', default=False)
    parser.add_argument('--summary', action='store', default=None)
    parser.add_argument('--stations', action='store', default=None)
    parser.add_argument('--tmpdir', action='store', default="/tmp")
    parser.add_argument('files', nargs='*')

    args = parser.parse_args()

    level = logging.WARNING
    if args.verbose:
        level = logging.DEBUG

    # f = OneLineExceptionFormatter('%(asctime)s|%(levelname)s|%(message)s|', '%m/%d/%Y %I:%M:%S %p')
    # root = logging.getLogger()
    # root.setLevel(level)
    # root.setFormatter(f)

    logging.basicConfig(level=level)


    os.umask(0o22)

    for f in args.files:
        (fn, ext) = os.path.splitext(os.path.basename(f))
        #logging.debug(f"processing: {f} fn={fn} ext={ext}")

        if ext == '.zip':
            zf = zipfile.ZipFile(f)
            for info in zf.infolist():
                try:
                    logging.debug(f"reading: {info.filename} from {f}")
                    data = zf.read(info.filename)
                    fd, path = tempfile.mkstemp(dir=args.tmpdir)
                    os.write(fd, data)
                    os.lseek(fd, 0, os.SEEK_SET)
                    file = os.fdopen(fd)
                except KeyError:
                    log.error(
                        f'ERROR: Did not find {info.filename} in zipe file {f}')
                else:
                    #logging.debug(f"processing: {info.filename} from {f}")
                    (bn, ext) = os.path.splitext(info.filename)
                    process(args, file, info.filename, f)
                    file.close()
                    os.remove(path)
        else:
            file = open(f, 'rb')
            process(args, file, f, None)
            file.close()

    if args.summary:
        update_summary(args, updated_stations)


if __name__ == "__main__":
    sys.exit(main())
