const fs = require('fs');
const path = require('path');
const lesionTypes = [];
const dir = './data';
const folders = ['train', 'test']
const imageFolders = ['benign', 'malignant'];

function prepareData(dir, numImages) {      

  folders.forEach(folder => {
    imageFolders.forEach((imageFolder, i) => {      
      const currentFolder = path.join(dir, folder, imageFolder);      
      let fileNames = fs.readdirSync(currentFolder);
      fileNames = fileNames.slice(0, numImages);      
      
      fileNames.forEach(fileName => {
        const filePath = path.join(currentFolder, fileName);    
        const image = fs.readFileSync(filePath);
        const lesionType = +(imageFolder === 'malignant'); 
        const fileNameWithoutExt = path.basename(fileName, path.extname(fileName));      
        const writePath = path.join(dir, 'prepared', folder, `${fileNameWithoutExt}_${lesionType}.jpg`);                    
        fs.writeFileSync(writePath, image, {encoding: 'base64'});               
      });
    });      
  }); 
};

prepareData(dir, 20);