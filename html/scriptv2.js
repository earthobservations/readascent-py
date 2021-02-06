var map = L.map('map', {
    'center': [15, 47],
    'zoom': 3,
    'layers': [
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            'attribution': 'Map data &copy; OpenStreetMap contributors'
        })
    ]
});

var datapath = 'https://radiosonde.mah.priv.at/data-dev/';
var summary = datapath + 'summary.geojson';

var geojsonMarkerOptions = {
    radius: 8,
    color: "#000",
    weight: 1,
    opacity: 1,
    fillOpacity: 0.8
};

// darkest blue = most recent
var base1 = new KolorWheel("#eeffff ");
var base2 = new KolorWheel("#eeffff ");
var maxHrs = 40;
marker_shades = {
    "BUFR": base1.abs("#330066 ", maxHrs),
    "netCDF": base2.abs("#336600 ", maxHrs)
}
path_colors = {
    "simulated": {
        color: 'DarkOrange'
    },
    "origin": {
        color: 'MediumBlue'
    }
}

function drawpath(geojson, l) {
    // record for later, and maybe a skewt on click
    //l.sourceTarget.feature.properties.ascents[0].data = geojson;
    path_source = geojson.properties.path_source;

    var lineCoordinate = [];
    for (var i in geojson.features) {
        var pointJson = geojson.features[i];
        var coord = pointJson.geometry.coordinates;
        lineCoordinate.push([coord[1], coord[0]]);
    }
    L.polyline(lineCoordinate, path_colors[path_source]).addTo(map);
}

function mouseover(l) {
    var ascents = l.target.ascents;

    for (const a of ascents) {
        if (!a.hasOwnProperty('data')) {
            var p = datapath + a.path;
            l.ascent = a;
            $.getJSON(p,
                (function(site) {
                    return function(geojson) {
                        site.ascent.data = geojson;
                        drawpath(geojson, site);
                    };
                }(l))
            );
        }
    }
}

function clicked(l) {
    let ctx = document.getElementById("chart");
    // debugger;
    // var f = l.sourceTarget.feature;
    var a = f.target.ascents;
    // if (f.properties.ascents[0].hasOwnProperty('data')) {
    //     alert("the ascent data is all loaded, but now it needs somebody more competent" +
    //         " to actually draw a SkewT diagram.")
    // }
    // else {
    //     var p = datapath + f.properties.ascents[0].path;
    //     alert("draw a skewT by loading JSON from: " + p);
    // }
}

function findBUFR(value, index, array) {
    return (value.source === "BUFR");
}

function findnetCDF(value, index, array) {
    return (value.source === "netCDF");
}

$.getJSON(summary, function(data) {
    L.geoJson(data, {

        pointToLayer: function(feature, latlng) {
            let now = Math.floor(Date.now() / 1000);
            let ascents = feature.properties.ascents;

            // prefer BUFR over netCDF
            // ascents are sorted descending by syn_timestamp
            // both netCDF either/or BUFR-derived ascents with same syn_timestamp
            // may be present.
            // BUFR-derived ascents are better quality data so prefer them.
            // we keep the netCDF-derived ascents of same timestamp around
            // to check how good the trajectory simulation is
            var newest_bufr = ascents.find(findBUFR);
            var newest_netcdf = ascents.find(findnetCDF);
            if (!newest_bufr && !newest_netcdf)
                return;

            if (newest_bufr && newest_netcdf &&
                (newest_bufr.syn_timestamp) ==
                (newest_netcdf.syn_timestamp)) {
                a = [newest_bufr, newest_netcdf];
            }
            else {
                if (newest_bufr)
                    a = [newest_bufr];
                else
                    a = [newest_netcdf];
            }
            var primary = a[0];
            var ts = primary.syn_timestamp;

            age = Math.round(Math.min((now - ts) / 3600, maxHrs - 1));
            age = Math.max(age, 0);
            // console.log(age, a);

            geojsonMarkerOptions.fillColor = marker_shades[primary.source].get(age).getHex();

            marker = L.circleMarker(latlng, geojsonMarkerOptions);
            marker.ascents = a;
            var content = "<b>" + feature.properties.name + "</b>" + "<b>  " +
                new Date(feature.properties.ascents[0].firstSeen * 1000).toLocaleString(undefined, {
                    year: 'numeric',
                    month: 'numeric',
                    day: 'numeric',
                    hour: '2-digit',
                    minute: '2-digit',
                }) + "</b><b> source: " + primary.source + "</b>";

            marker.bindTooltip(content).openTooltip()
                .on('click', clicked)
                .on('mouseover', mouseover);

            return marker;
        }
    }).addTo(map);
});
