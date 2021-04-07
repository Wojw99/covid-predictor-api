const express = require('express');
const app = express();
const csv = require('csv-parser');
const fs = require('fs');


// - - - - - GET main route - - - - - */
app.get('/', function (request, response) {
    response.send('Hello World!');
});

// - - - - - GET for all predictions - - - - - */
app.get('/api/predictions', function (request, response) {
    var predictions = []

    fs.createReadStream('data/data_predicted.csv')
    .pipe(csv())
    .on('open', () => {
        console.log('start reading');
    })
    .on('data', (row) => {
        predictions.push(row);
    })
    .on('end', () => {
        console.log('end reading');
        response.send(predictions);
    });    
});

// - - - - - - - RUN SERVER - - - - -  - - 
app.listen(3000, function(){
    console.log('Listening on port 3000...');
});