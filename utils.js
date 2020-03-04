const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');

function loadImage(filePath) {  
  return fs.readFileSync(filePath);
}

function createTensorFromImage(image, width = 224, height = 224) {
  return tf.node.decodeImage(image)
      .resizeNearestNeighbor([width,height])
      .toFloat()
      .div(tf.scalar(255.0))
      .expandDims();    
}

module.exports = {
  loadImage,
  createTensorFromImage
}


