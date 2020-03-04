const path = require('path');
const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');
const {loadImage, createTensorFromImage} = require('./utils');

async function makePrediction() {
  const model = await tf.loadLayersModel('file://models/mymodel4/model.json');

  const predictFolder = path.join('./data', 'predict');
  let fileNames = fs.readdirSync(predictFolder);    
  fileNames.forEach(async fileName => {
    const filePath = path.join(predictFolder, fileName);    
    const image = fs.readFileSync(filePath);
    const tensor = createTensorFromImage(image, 96, 96);    
    const predictions = await model.predict(tensor).data();    
    const lesionPrediction = Array.from(predictions);
    console.log(lesionPrediction);
    console.log(`Malignant: ${lesionPrediction[1]>lesionPrediction[0]} : ${fileName}`);
  }); 
}

makePrediction();
