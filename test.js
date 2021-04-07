const EventEmitter = require('events');
const Logger = require('./logger.js');
const fs = require('fs');

const logger = new Logger();

// Register a listener
logger.on('messageLogged', function(args){
    console.log(`Hello Word!, ${args.id}, ${args.url}`);
});

// logger.log('message');


// - - - - - CSV - - - - - 
var row = {
    'Country/Region': 'Tajikistan ',
    Lat: '38.861034000000004',
    Long: '71.276093',
    '2/10/20': '0',
    '2/11/20': '0',
    '2/12/20': '0',
    '2/13/20': '0',
    '2/14/20': '0',
    '2/15/20': '0'
};

console.log(row.Lat);
console.log(row['Country/Region']);
row.forEach(element => {
    console.log(element);
});