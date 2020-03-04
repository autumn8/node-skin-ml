const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');
const argv = require('yargs').argv;
const LESION_TYPE = require('./lesionType');
const {createTensorFromImage}= require('./utils');
const TRAINING_DATA_PATH = './data/prepared/train';
const TESTING_DATA_PATH = './data/prepared/test';
const express = require('express');
const app = express();
const http = require('http').createServer(app);
const io = require('socket.io')(http);
const history = [];

app.use(express.static('public'));

app.get('/', function(req, res){
  res.sendFile(__dirname + '/index.html');
});

io.on('connection', function(socket){
  console.log('a user connected');  
  io.emit('history', history);   
});

http.listen(3000, function(){
  console.log('listening on *:3000');
});


const model = require('./model');

//TODO add tf.tidy wrappers

function getData(dir) {
  const features = [];
  const labels = [];
  let fileNames = fs.readdirSync(dir);
  tf.util.shuffle(fileNames);
  fileNames.forEach(async fileName => {
    const filePath = path.join(dir, fileName);    
    const image =  fs.readFileSync(filePath);
    const tensor = createTensorFromImage(image, 96, 96);  
    features.push(tensor);
    const isMalignant = fileName.endsWith(`_${LESION_TYPE.MALIGNANT}.jpg`);
    labels.push(isMalignant);    
  });

  return {
    features: tf.concat(features),
    labels: tf.oneHot(tf.tensor1d(labels, 'int32'), 2).toFloat()
  }
}

async function trainModel() {
  const trainingData = getData(TRAINING_DATA_PATH);
  const testingData = getData(TESTING_DATA_PATH); 

  model.summary();
  
  
  const modelTrainingResult = await model.fit(trainingData.features, trainingData.labels, {
    epochs: 2,
    batchSize: 32,
    shuffle: true,
    validationSplit: 0.6,
    callbacks: {
      onEpochEnd: (epoch, log) => {
        console.log(epoch);
        history.push(log);
        console.log(history);  
        io.emit('history', history);         
      }
    }
  });

  console.log('training complete');
  console.log(modelTrainingResult);
  const modelEvaluation = model.evaluate(testingData.features, testingData.labels);

  console.log(
      `\nEvaluation result:\n` +
      `  Loss = ${modelEvaluation[0].dataSync()[0].toFixed(3)}; `+
      `Accuracy = ${modelEvaluation[1].dataSync()[0].toFixed(3)}`);
  
  let modelFileName = argv.modelFileName || 'savedModel';

  await model.save(`file://./models/${modelFileName}`);
  console.log(`Saved model to path: ./models/${modelFileName}`);

}

trainModel();