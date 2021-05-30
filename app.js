const express = require('express');
const app = express();
const csv = require('csv-parser');
const fs = require('fs');

var dataPredicted = [];

readDataPredicted();

// read .csv file and fill in the dataPredicted list with data
async function readDataPredicted() {
    fs.readFile('data/data_predicted.csv', 'utf-8', (error, data) => {
        if (error) {
            console.log(error);
            return;
        }

        var lines = data.split('\n');
        var headers = lines[0].split(',');

        for (var i = 1; i < lines.length; i++) {
            var elements = lines[i].split(',');
            var region = elements[0];
            var listOfOutputs = [];

            // fill in the list of outputs
            for (var j = 1; j < elements.length - 4; j++) {
                listOfOutputs.push({
                    date: headers[j],
                    cases: elements[j],
                });
            }

            dataPredicted.push({
                region: region,
                outputs: listOfOutputs,
            });
        }
    });
}



// - - - - - GET main route - - - - - */
app.get('/', function (request, response) {
    response.send('Hello World!');
});

// - - - - - GET for all predictions - - - - - */
app.get('/api/predictions', function (request, response) {
    response.send(dataPredicted);
});

// - - - - - GET test predictions - - - - - */
app.get('/api/test/predictions', function (request, response) {
    var predictions = [
        {
            region: 'Tajikistan',
            outputs: [
                {
                    date: '2/10/20',
                    cases: '0'
                },
                {
                    date: '3/10/20',
                    cases: '2'
                },
                {
                    date: '4/10/20',
                    cases: '230'
                },
                {
                    date: '5/10/20',
                    cases: '2444'
                },
            ]
        },
        {
            region: 'Warsaw ',
            outputs: [
                {
                    date: '2/10/20',
                    cases: '0'
                },
                {
                    date: '3/10/20',
                    cases: '2'
                },
                {
                    date: '4/10/20',
                    cases: '230'
                },
                {
                    date: '5/10/20',
                    cases: '2444'
                },
            ]
        }
    ];
    response.send(predictions);
});

// - - - - - - - RUN SERVER - - - - -  - - 
app.listen(3000, function () {
    console.log('Listening on port 3000...');
});