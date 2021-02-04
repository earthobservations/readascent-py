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


function onEachFeature(feature, layer) {
    console.log("onEachFeature", feature, layer);
}

function do_something_with_data(data, l) {
    debugger;
    l.sourceTarget.feature.properties.ascents[0].data = data;
    console.log(data, l);
}

function mouseover(l) {
    var f = l.sourceTarget.feature;
    var p = datapath + f.properties.ascents[0].path;
    //debugger;

    if (!f.properties.ascents[0].hasOwnProperty('data')) {

        $.getJSON(p,
            (function(thisl) {
                return function(data) {
                    do_something_with_data(data, thisl);
                    // Break the closure over `i` via the parameter `thisi`,
                    // which will hold the correct value from *invocation* time.
                };
            }(l)) // calling the function with the current value
        );
    }
    else {
        console.log("already loaded", f.properties.ascents[0].data);
    }

    // $.getJSON(
    //     p,
    //     (function(thisl) {
    //         return function(data) {
    //             do_something_with_data(data, thisl);
    //             // Break the closure over `i` via the parameter `thisi`,
    //             // which will hold the correct value from *invocation* time.
    //         };
    //     }(l)) // calling the function with the current value
    // );

    //   $.getJSON(p, { blah: "FASEL"})
    //   .done(function( json ) {
    //       debugger;
    //     console.log( "JSON Data: ");
    //   })
    //   .fail(function( jqxhr, textStatus, error ) {
    //     var err = textStatus + ", " + error;
    //     console.log( "Request Failed: " + err );
    // });


    // $.getJSON(p, { name: "John", time: "2pm" }, function(geojson, l) {
    //     debugger;
    //     var lineCoordinate = [];
    //     for(var i in geojson.features){
    //       var pointJson = geojson.features[i];
    //       var coord = pointJson.geometry.coordinates;
    //       lineCoordinate.push([coord[1],coord[0]]);
    //     }
    //     //debugger;
    //     L.polyline(lineCoordinate, {color: 'red'}).addTo(map);
    // });

}

function clicked(l) {
    //debugger;
    console.log("clicked", l);
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
        //, onEachFeature: onEachFeature

    }).addTo(map);
});
