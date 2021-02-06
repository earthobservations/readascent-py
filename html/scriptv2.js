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
    l.sourceTarget.feature.properties.ascents[0].data = geojson;
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
    var f = l.sourceTarget.feature;
    var p = datapath + f.properties.ascents[0].path;

    if (!f.properties.ascents[0].hasOwnProperty('data')) {
        $.getJSON(p,
            (function(site) {
                return function(geojson) {
                    drawpath(geojson, site);
                };
            }(l))
        );
    } // else already loaded
}

function clicked(l) {
    let ctx = document.getElementById("chart");
    // debugger;
    var f = l.sourceTarget.feature;
    // if (f.properties.ascents[0].hasOwnProperty('data')) {
    //     alert("the ascent data is all loaded, but now it needs somebody more competent" +
    //         " to actually draw a SkewT diagram.")
    // }
    // else {
    //     var p = datapath + f.properties.ascents[0].path;
    //     alert("draw a skewT by loading JSON from: " + p);
    // }
}

$.getJSON(summary, function(data) {
    L.geoJson(data, {

        pointToLayer: function(feature, latlng) {
            let now = Math.floor(Date.now() / 1000);
            let ascents = feature.properties.ascents

            var newest_ascent = ascents[0];
            var ts  = ascents[0].syn_timestamp;

            // prefer BUFR over netCDF
            // ascents are sorted descending by syn_timestamp
            // both netCDF either/or BUFR-derived ascents with same syn_timestamp
            // may be present.
            // BUFR-derived ascents are better quality data so prefer them.
            // we keep the netCDF-derived ascents of same timestamp around
            // to check how good the trajectory simulation is
            if ((newest_ascent.source === "netCDF") &&  (ascents.length > 1)) {
                second = ascents[1];
                if ((second.source === "BUFR") && (second.syn_timestamp == ts)) {
                    newest_ascent = second;
                }
            }

            ts = newest_ascent.firstSeen;
            age = Math.round(Math.min((now - ts) / 3600, maxHrs - 1));
            age = Math.max(age, 0);
            source = newest_ascent.source; // "netCDF" or "BUFR"
            path_source = newest_ascent.path_source; // "simulated" or "origin"

            console.log(age, source, path_source);

            geojsonMarkerOptions.fillColor = marker_shades[source].get(age).getHex();

            marker = L.circleMarker(latlng, geojsonMarkerOptions);

            var content = "<b>" + feature.properties.name + "</b>" + "<b>  " +
                new Date(feature.properties.ascents[0].firstSeen * 1000).toLocaleString(undefined, {
                    year: 'numeric',
                    month: 'numeric',
                    day: 'numeric',
                    hour: '2-digit',
                    minute: '2-digit',
                }) + "</b><b> source: " + source + "</b>";

            marker.bindTooltip(content).openTooltip()
                .on('click', clicked)
                .on('mouseover', mouseover);

            return marker;
        }
    }).addTo(map);
});
