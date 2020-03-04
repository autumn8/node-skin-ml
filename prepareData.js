const fs = require('fs');
const argv = require('yargs').argv;
const path = require('path');
const folders = ['train', 'test']
const imageFolders = ['benign', 'malignant'];

const dir = argv.dir || './data';

folders.forEach(folder => {  
  imageFolders.forEach((imageFolder, i) => {      
    const currentFolder = path.join(dir, folder, imageFolder);      
    let fileNames = fs.readdirSync(currentFolder);
    console.log(fileNames.length);
    if (argv.numImages) fileNames = fileNames.slice(0, argv.numImages);      
    console.log(fileNames.length);
    
    fileNames.forEach(fileName => {
      const filePath = path.join(currentFolder, fileName);    
      const image = fs.readFileSync(filePath);
      const lesionType = +(imageFolder === 'malignant'); 
      const fileNameWithoutExt = path.basename(fileName, path.extname(fileName));      
      const fileNameWithLesionType = `${fileNameWithoutExt}_${lesionType}.jpg`;
      const writePath = path.join(dir, 'prepared', folder, fileNameWithLesionType);                    
      fs.writeFileSync(writePath, image, {encoding: 'base64'});               
    });
  });      
}); 
