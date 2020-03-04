const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');
const argv = require('yargs').argv;
const LESION_TYPE = require('./lesionType');
const {loadImage, createTensorFromImage}= require('./utils');
const TRAINING_DATA_PATH = './data/prepared/train';
const TESTING_DATA_PATH = './data/prepared/test';

const model = require('./model');

//TODO add tf.tidy wrappers

function getData(dir) {
  const features = [];
  const labels = [];
  let fileNames = fs.readdirSync(dir);
  tf.util.shuffle(fileNames);
  fileNames.forEach(async fileName => {
    const filePath = path.join(dir, fileName);    
    const image = await loadImage(filePath);
    const tensor = createTensorFromImage(image, 96, 96);  
    features.push(tensor);
    const isMalignant = fileName.endsWith(`_${LESION_TYPE.MALIGNANT}.jpg`);
    labels.push(isMalignant);    
  });

  return {
    features,
    labels
  }
}

async function trainModel() {
  const trainingData = getData(TRAINING_DATA_PATH);
  const testData = getData(TESTING_DATA_PATH);

  const trainingDataTensor = {
    images: tf.concat(trainingData.features),
    labels: tf.oneHot(tf.tensor1d(trainingData.labels, 'int32'), 2).toFloat()
  } 
  
  const testingDataTensor = {
    images: tf.concat(testData.features),
    labels: tf.oneHot(tf.tensor1d(testData.labels, 'int32'), 2).toFloat()
  }

  model.summary();
  
  const history = [];
  const modelTrainingResult = await model.fit(trainingDataTensor.images, trainingDataTensor.labels, {
    epochs: 100,
    batchSize: 32,
    shuffle: true,
    validationSplit: 0.6,
    callbacks: {
      onEpochEnd: (epoch, log) => {
        history.push(log);
        console.log(history);  
        //tfvis.show.history(surface, history, ['loss', 'acc']);
      }
    }
  });


  const modelEvaluation = model.evaluate(testingDataTensor.images, testingDataTensor.labels);

  console.log(
      `\nEvaluation result:\n` +
      `  Loss = ${modelEvaluation[0].dataSync()[0].toFixed(3)}; `+
      `Accuracy = ${modelEvaluation[1].dataSync()[0].toFixed(3)}`);
  
  let modelFileName = argv.modelFileName || 'savedModel';

  await model.save(`file://./models/${modelFileName}`);
  console.log(`Saved model to path: ./models/${modelFileName}`);

}

trainModel();