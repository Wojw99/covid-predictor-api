const EventEmitter = require('events');
const Logger = require('./logger.js');

const logger = new Logger();

// Register a listener
logger.on('messageLogged', function(args){
    console.log(`Hello Word!, ${args.id}, ${args.url}`);
});

logger.log('message');
