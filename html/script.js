var map = L.map('map', {
    'center': [15, 47],
    'zoom': 3,
    'layers': [
        L.tileLayer('http://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            'attribution': 'Map data &copy; OpenStreetMap contributors'
        })
    ]
});

var url = 'https://radiosonde.mah.priv.at/data/summary.geojson';

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


function onEachFeature(feature, layer) {
    console.log("onEachFeature", feature, layer);
    var popupContent = "<b>" + feature.properties.name + "</b>" + "<b>  " +
        new Date(feature.properties.ascents[0].firstSeen * 1000).toLocaleString(undefined, {
            year: 'numeric',
            month: 'numeric',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
        }) + "</b>";

    layer.bindPopup(popupContent);
}


$.getJSON(url, function(data) {
    L.geoJson(data, {

        pointToLayer: function(feature, latlng) {
            now = Math.floor(Date.now() / 1000);
            ts = feature.properties.ascents[0].firstSeen;
            age = Math.round(Math.min((now - ts) / 3600, maxHrs - 1));
            geojsonMarkerOptions.fillColor = target.get(age).getHex();

            //L.bindPopup(feature.properties.name);

            return L.circleMarker(latlng, geojsonMarkerOptions);
        },
        onEachFeature: onEachFeature

    }).addTo(map);
});
