#  This program was automatically generated with bufr_dump -Dpython
#  Using ecCodes version: 2.19.1
#
# bufr_dump -Dpython A_IUSD02LOWM210300_C_EDZW_20210121040000_59339751.bin >dump.py
# and hand edited.

import traceback
import sys
from eccodes import *
import argparse
import pprint
# import geojson
# import geobuf
# import czml3
import logging

def bufr_decode(input_file, args):
    f = open(input_file, 'rb')
    ibufr = codes_bufr_new_from_file(f)
    codes_set(ibufr, 'unpack', 1)

    skippedHdrs = 0

    header = dict()
    arraykeys = ['delayedDescriptorReplicationFactor',
                 'extendedDelayedDescriptorReplicationFactor',
                 'unexpandedDescriptors']

    for k in arraykeys:
        try:
            header[k] = list(codes_get_array(ibufr, k))
        except Exception as e:
            logging.error(f"array key in header key={k} e={e}")
            skippedHdrs += 1



    num_samples = header['extendedDelayedDescriptorReplicationFactor'][0]
    logging.debug(f"num_samples={num_samples}")

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
            skippedHdrs += 1

    keys = ['timePeriod', 'extendedVerticalSoundingSignificance',
            'pressure', 'nonCoordinateGeopotentialHeight',
            'latitudeDisplacement', 'longitudeDisplacement',
            'airTemperature', 'dewpointTemperature',
            'windDirection', 'windSpeed']

    samples = []
    skippedSamples = 0
    for i in range(1, num_samples+1):
        sample = dict()
        for k in keys:
            name = f"#{i}#{k}"
            try:
                sample[k] = codes_get(ibufr, name)
            except Exception as e:
                logging.debug(f"sample={i} key={k} e={e}, setting to None")
                sample[k] = None
        # call BS on bogus values
        if float(sample['airTemperature']) < -273 or float(sample['dewpointTemperature']) < -273:
            skippedSamples += 1
            continue
        samples.append(sample)
    logging.debug(f"samples used={len(samples)} skipped samples={skippedSamples} skipped header keys={skippedHdrs}")

    header['samples'] = samples

    if args.geojson or args.geobuf:
        gen_geojson(header, args.geobuf)
    elif args.czml:
        gen_czml(header)
    else:
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(header)

    codes_release(ibufr)
    f.close()

def gen_geojson(header, geobuf):
    logging.warning(f"not implemented yet")

def gen_czml(header):
    logging.warning(f"not implemented yet")



def main():
    parser = argparse.ArgumentParser(description='decode radiosonde BUFR report',
                                    add_help=True)
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parser.add_argument('--geojson', action='store_true', default=False)
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
        try:
            logging.debug(f"processing {f}")
            bufr_decode(f, args)
        except CodesInternalError as err:
            traceback.print_exc(file=sys.stderr)
    logging.debug('Finished')


if __name__ == "__main__":
    sys.exit(main())
