#  This program was automatically generated with bufr_dump -Dpython
#  Using ecCodes version: 2.19.1
#
# bufr_dump -Dpython A_IUSD02LOWM210300_C_EDZW_20210121040000_59339751.bin >dump.py
# and hand edited.

import traceback
import sys
from eccodes import *
import argparse
import json
import geojson
import geobuf
# import czml3
import logging
import datetime
import os

def bufr_decode(input_file, args):
    f = open(input_file, 'rb')
    ibufr = codes_bufr_new_from_file(f)
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
            logging.error(f"array key in header key={k} e={e}")
            missingHdrKeys += 1

    num_samples = header['extendedDelayedDescriptorReplicationFactor'][0]
    logging.info(f"num_samples={num_samples}")

    scalarkeys = [
        'edition',
        'masterTableNumber',
        'bufrHeaderCentre',
        'bufrHeaderSubCentre',
        'updateSequenceNumber',
        'dataCategory',
        'internationalDataSubCategory',
        'dataSubCategory',
        'masterTablesVersionNumber',
        'localTablesVersionNumber',
        'typicalYear',
        'typicalMonth',
        'typicalDay',
        'typicalHour',
        'typicalMinute',
        'typicalSecond',
        'numberOfSubsets',
        'observedData',
        'compressedData',
        'radiosondeSerialNumber',
        'radiosondeAscensionNumber',
        'radiosondeReleaseNumber',
        'observerIdentification',
        'radiosondeCompleteness',
        'radiosondeConfiguration',
        'correctionAlgorithmsForHumidityMeasurements',
        'radiosondeGroundReceivingSystem',
        'radiosondeOperatingFrequency',
        'balloonManufacturer',
        'weightOfBalloon',
        'amountOfGasUsedInBalloon',
        'pressureSensorType',
        'temperatureSensorType',
        'humiditySensorType',
        'geopotentialHeightCalculation',
        'softwareVersionNumber',
        'reasonForTermination',
        'blockNumber',
        'stationNumber',
        'radiosondeType',
        'solarAndInfraredRadiationCorrection',
        'trackingTechniqueOrStatusOfSystem',
        'measuringEquipmentType',
        'timeSignificance',
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
        'text']

    for k in scalarkeys:
        try:
            header[k] = codes_get(ibufr, k)
        except Exception as e:
            logging.error(f"scalar hdr key={k} e={e}")
            missingHdrKeys += 1

    keys = ['timePeriod', 'extendedVerticalSoundingSignificance',
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
    f.close()
    return header, samples


def extract_header(args, h, samples):
    takeoff = datetime.datetime(year=h['year'],
                                   month=h['month'],
                                   day=h['day'],
                                   minute=h['minute'],
                                   hour=h['hour'],
                                   second=h['second'],
                                   tzinfo=None)
    properties = {
        "firstSeen": takeoff.isoformat(),
        "lat": h['latitude'],
        "lon": h['longitude'],
        "elevation": h['height'],  # ?
        "station_elevation": h['heightOfStationGroundAboveMeanSeaLevel'], # ?
        "baro_elevation": h['heightOfBarometerAboveMeanSeaLevel'], # ?
        "sonde_serial": h['radiosondeSerialNumber'],
        "sonde_frequency":  h['radiosondeOperatingFrequency'],
        "sonde_type":  h['radiosondeType']
    }
    fc = geojson.FeatureCollection([])
    fc.properties = properties

    # add name "Location time" or geocode location
    # add wmo_id

    lat_t = fc.properties['lat']
    lon_t = fc.properties['lon']
    firstSeen = fc.properties['firstSeen']
    previous_elevation = fc.properties['elevation'] - args.hstep

    for s in samples:
        lat = lat_t + s['latitudeDisplacement']
        lon = lon_t + s['longitudeDisplacement']
        ele = s['nonCoordinateGeopotentialHeight']
        if ele < previous_elevation + args.hstep:
            continue
        previous_elevation = ele
        delta = datetime.timedelta(seconds=s['timePeriod'])
        sampleTime = takeoff + delta
        properties = {
            "time": sampleTime.timestamp(),
            "gpheight": ele,
            "temp": s['airTemperature'],
            "dewpoint": s['dewpointTemperature'],
            "pressure": s['pressure'],
            "wdir": s['windDirection'],
            "wspeed": s['windSpeed']
        }
        f = geojson.Feature(geometry=geojson.Point((lon, lat, ele)),
                            properties=properties)
        fc.features.append(f)
    fc.properties['lastSeen'] = sampleTime.isoformat()
    return fc

def gen_czml(fc):
    logging.warning(f"not implemented yet")

def main():
    parser = argparse.ArgumentParser(description='decode radiosonde BUFR report',
                                     add_help=True)
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parser.add_argument('--orig', action='store_true', default=False)
    parser.add_argument('--hstep', action='store', type=int, default=0)
    parser.add_argument('--destdir', action='store', default=".")
    parser.add_argument('--geojson', action='store_true', default=False)
    parser.add_argument('--dump-geojson', action='store_true', default=False)
    parser.add_argument('--geobuf', action='store_true', default=False)
    parser.add_argument('--czml', action='store_true', default=False)
    parser.add_argument('--reportevery', action='store', default=1)
    parser.add_argument('files', nargs='*')

    args = parser.parse_args()

    level = logging.WARNING
    if args.verbose:
        level = logging.DEBUG
    logging.basicConfig(level=level)

    for f in args.files:
        (fn, ext) = os.path.splitext(os.path.basename(f))
        try:
            logging.debug(f"processing: {f}")
            (h, s) = bufr_decode(f, args)
        except CodesInternalError as err:
            traceback.print_exc(file=sys.stderr)

        h['samples'] = s
        if args.orig:
            print(h)

        fc = extract_header(args, h, s)

        if args.geojson:
            dest = f'{args.destdir}/{fn}.geojson'
            logging.debug(f'writing {dest}')
            with open(dest, 'wb') as gjfile:
                gj = geojson.dumps(fc, indent=4)
                gjfile.write(gj.encode("utf8"))

        if args.geobuf:
            dest = f'{args.destdir}/{fn}.geobuf'
            logging.debug(f'writing {dest}')
            with open(dest, 'wb') as gbfile:
                gb = geobuf.encode(fc)
                gbfile.write(gb)

        if args.czml:
            gen_czml(f)

        if args.dump_geojson:
            print(gj)

    logging.debug('Finished')


if __name__ == "__main__":
    sys.exit(main())
