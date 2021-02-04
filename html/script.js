var map = L.map('map', {
    'center': [15, 47],
    'zoom': 3,
    'layers': [
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            'attribution': 'Map data &copy; OpenStreetMap contributors'
        })
    ]
});

var url = 'https://radiosonde.mah.priv.at/data/summary.geojson';
var datapath = 'https://radiosonde.mah.priv.at/data/';

var geojsonMarkerOptions = {
    radius: 8,
    color: "#000",
    weight: 1,
    opacity: 1,
    fillOpacity: 0.8
};

// darkest blue = most recent
var base = new KolorWheel("#eeffff ");
var maxHrs = 40;
var target = base.abs("#330066 ", maxHrs);


function drawpath(geojson, l) {
    // record for later, and maybe a skewt on click
    l.sourceTarget.feature.properties.ascents[0].data = geojson;

    var lineCoordinate = [];
    for (var i in geojson.features) {
        var pointJson = geojson.features[i];
        var coord = pointJson.geometry.coordinates;
        lineCoordinate.push([coord[1], coord[0]]);
    }
    L.polyline(lineCoordinate, {
        color: 'red'
    }).addTo(map);
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
    if (f.properties.ascents[0].hasOwnProperty('data')) {
        alert("the ascent data is all loaded, but now it needs somebody more competent" +
            " to actually draw a SkewT diagram.")
    } else {
        var p = datapath + f.properties.ascents[0].path;
        alert("draw a skewT by loading JSON from: " + p);
    }
}

$.getJSON(url, function(data) {
    L.geoJson(data, {

        pointToLayer: function(feature, latlng) {
            now = Math.floor(Date.now() / 1000);
            ts = feature.properties.ascents[0].firstSeen;
            age = Math.round(Math.min((now - ts) / 3600, maxHrs - 1));
            geojsonMarkerOptions.fillColor = target.get(age).getHex();

            marker = L.circleMarker(latlng, geojsonMarkerOptions);

            var content = "<b>" + feature.properties.name + "</b>" + "<b>  " +
                new Date(feature.properties.ascents[0].firstSeen * 1000).toLocaleString(undefined, {
                    year: 'numeric',
                    month: 'numeric',
                    day: 'numeric',
                    hour: '2-digit',
                    minute: '2-digit',
                }) + "</b>";

            marker.bindTooltip(content).openTooltip()
                .on('click', clicked)
                .on('mouseover', mouseover);

            return marker;
        }
    }).addTo(map);
});
