const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

const model = require('./model');


function getData(dir) {
  const features = [];
  const labels = [];
  let fileNames = fs.readdirSync(dir);
  fileNames.forEach(fileName => {
    const filePath = path.join(dir, fileName);    
    const image = fs.readFileSync(filePath);
    const imageTensor = tf.node.decodeImage(image)
      .resizeNearestNeighbor([224,224])
      .toFloat()
      .div(tf.scalar(255.0))
      .expandDims();
    
    features.push(imageTensor);
    
    const isMalignant = fileName.endsWith('_1.jpg');
    labels.push(isMalignant);    
  });

  return {
    features,
    labels
  }
}

async function trainModel() {
  const data = getData('./data/prepared/train');

  const trainingData = {
    images: tf.concat(data.features),
    labels: tf.oneHot(tf.tensor1d(data.labels, 'int32'), 2).toFloat()
  }  

  model.summary();
  
  console.log(trainingData);

  const validationSplit = 0.15;

  await model.fit(trainingData.images, trainingData.labels, {
    epochs: 100,
    batchSize: 32,
    validationSplit: 0.6
  });

}

trainModel();



