const fs = require('fs');
const csv = require('csv-parser');

const results = [];

fs.createReadStream('../out_files/results.out')
  .pipe(csv())
  .on('data', (data) => results.push(data))
  .on('end', () => {
    console.log('CSV parsed:', results);
    // You can now process, filter, or save the results as needed
    // For example, save as JSON:
    fs.writeFileSync('./data_files/results.json', JSON.stringify(results, null, 2));
  });