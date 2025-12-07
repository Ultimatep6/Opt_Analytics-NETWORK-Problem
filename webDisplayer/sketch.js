var gridHeight = 400
var gridWidth = 400

var objSize = 15;

// Make a Grid
var grid = new Grid();

function getConnections(grid) {
	for (let keyA in grid.nodes) {
		let nodeA = grid.nodes[keyA];
		for (let keyB in grid.nodes) {
			let nodeB = grid.nodes[keyB];
			// Check if nodeA and nodeB are adjacent
			if ((nodeA.i === nodeB.i && Math.abs(nodeA.j - nodeB.j) === 1) ||
				(nodeA.j === nodeB.j && Math.abs(nodeA.i - nodeB.i) === 1)) {
				// Create a new connection and add it to the list
				var connection = new Connection(nodeA, nodeB, grid);
				grid.addConnection(connection);
			}
		}
	}
}

// Use p5.js's preload to load settings before setup
let settings;

function preload() {
	settings = loadJSON('../out_files/settings.json');
	results = loadJSON('./data_files/results.json');
}

function setup() {
	createCanvas(windowWidth, windowHeight);
	// console.log('Setup complete');



	// Initialize grid with settings
	cols = settings.cols;
	rows = settings.rows;

	var n_resource = settings.n_resources;
	var n_products = settings.n_products;

	var products = [];
	for (var i = 0; i < n_products; i++) {
		grid.addMaterialColor(`product_${i}`, color(random(255), random(255), random(255)));
		products.push(`products_${i}`)
	}
	var resources = [];
	for (var i = 0; i < n_resource; i++) {
	    grid.addMaterialColor(`resource_${i}`, color(random(255), random(255), random(255)));
		resources.push(`resource_${i}`)
	}

	// Making a 2D array
	for (var i = 0; i < cols; i++) {
		for (var j = 0; j < rows; j++) {
			grid.addNode(new Node(i, j));
		}
	}

	// Get the connections
	getConnections(grid);
	// console.log(grid.connections);

	
	// Get Sinkholes
	let factoryPositions = Array.isArray(settings.factory_positions)
		? settings.factory_positions
		: Object.values(settings.factory_positions);
	
	// Get Factories
	let factory_D_ARR = settings.factory_demand_matrix;
	let factory_S_ARR = settings.factory_supply_matrix;

	console.log("Demand Arr:", factory_D_ARR)
	console.log("Supply Arr:", factory_S_ARR)

	factoryPositions.forEach((f, idx) => {
		console.log(`Factory ${idx}:, ${f[0]}, ${f[1]}`);
		var fact = new Factory(f[0], f[1], objSize)
		grid.addFactory(fact);

		// Add demand and supply for each product for this factory
		for (let p = 0; p < n_products; p++) {
			let productKey = `product_${p + 1}`; // product_1, product_2, ...
			let factoryKey = `fact_${idx + 1}`; // factory_1, factory_2, ...
			let matrixKey = `('${productKey}', '${factoryKey}')`;
			let demand = factory_D_ARR[matrixKey] || 0;

			console.log(matrixKey)
			fact.addDemand(productKey, demand);

			let supply = factory_S_ARR[matrixKey] || 0;
			fact.addSupply(productKey, supply);
		}

		for (let r = 0; r < n_resource; r++) {
			let resourceKey = `resource_${r + 1}`; // resource_1, resource_2, ...
			let factoryKey = `fact_${idx + 1}`; // factory_1, factory_2, ...
			let matrixKey = `('${resourceKey}', '${factoryKey}')`;
			let demand = factory_D_ARR[matrixKey] || 0;
			fact.addDemand(resourceKey, demand);

			let supply = factory_S_ARR[matrixKey] || 0;
			fact.addSupply(resourceKey, supply);
		}
	});

	// Get Sinkholes
	let sinkholePositions = Array.isArray(settings.sinkhole_positions)
		? settings.sinkhole_positions
		: Object.values(settings.sinkhole_positions);

	let sinkholeDemandMatrix = settings.sinkhole_demand_matrix;
	sinkholePositions.forEach((s, idx) => {
		console.log(`Sinkhole ${idx}: ${s[0]}, ${s[1]}`);
		var sink = new Sinkhole(s[0], s[1]);
		grid.addSink(sink);

		// Add demand for each product for this sinkhole
		for (let p = 0; p < n_products; p++) {
			let productKey = `product_${p + 1}`; // product_1, product_2, ...
			let sinkKey = `sink_${idx + 1}`;     // sink_1, sink_2, ...
			let matrixKey = `('${productKey}', '${sinkKey}')`;
			let demand = sinkholeDemandMatrix[matrixKey] || 0;
			sink.addDemand(productKey, demand);
		}
	});
	
	// Get Sources
	let sourcePositions = Array.isArray(settings.source_positions)
		? settings.source_positions
		: Object.values(settings.source_positions);


		
	let sourceSupplyMatrix = settings.source_supply_matrix;
	sourcePositions.forEach((s, idx) => {
		console.log(`Source ${idx}: ${s[0]}, ${s[1]}`);
		var source = new Source(s[0], s[1]);
		grid.addSource(source);


		// Add supply for each resource for this source
		for (let r = 0; r < n_resource; r++) {
			let resourceKey = `resource_${r + 1}`; // resource_1, resource_2, ...
			let sourceKey = `source_${idx + 1}`;     // source_1, source_2, ...
			let matrixKey = `('${resourceKey}', '${sourceKey}')`;
			let supply = sourceSupplyMatrix[matrixKey] || 0;
			source.addSupply(resourceKey, supply);
		}
	});

	// // Draw a factory for testing
	// grid.addFactory(new Factory(1, 1, 100, 150, objSize));
	
	
	// // Draw a sinkhole for testing
	// grid.addSink(new Sinkhole(3, 2, 80, objSize));	

	// // Draw a source for testing
	// var source = new Source(2, 4, 80, objSize);
	// grid.addSource(source);

	// // Add a material flow to a connection for testing
	// grid.addMaterialColor('water', color(0, 0, 255));
	
}

function draw() {
// 	// put drawing code here
// 	// background(220,0,100);
	for (let key in grid.nodes) {
		grid.nodes[key].show();
	}

	for (let key in grid.connections) {
		grid.showDefault(grid.connections[key]);
	}

	for (let key in grid.factories) {
		grid.nodes[key].hide();
		grid.factories[key].show();
	}

	for (let key in grid.sinks) {
		grid.nodes[key].hide();
		grid.sinks[key].show();
	}

	for (let key in grid.sources) {
		grid.nodes[key].hide();
		grid.sources[key].show();
	}

// 	// Find a connection to test
// 	let testConnectionKey = Object.keys(grid.connections)[0];
// 	let testConnection = grid.connections[testConnectionKey];
// 	// Show flow on that connection
// 	grid.showFlow(testConnection, 'water');

}

function mousePressed() {
    for (let key in grid.nodes){
		if (key in grid.factories) {
			if (grid.factories[key].isClicked(mouseX, mouseY)) {
				grid.factories[key].selected = !grid.factories[key].selected; // Toggle selection
				grid.factories[key].logFactory(); // Optional: log position
			}
		}

		else if (key in grid.sinks) {
			if (grid.sinks[key].isClicked(mouseX, mouseY)) {
				grid.sinks[key].selected = !grid.sinks[key].selected; // Toggle selection
				grid.sinks[key].logSink(); // Optional: log position
			}
		}

		else if (key in grid.sources) {
			if (grid.sources[key].isClicked(mouseX, mouseY)) {
				grid.sources[key].selected = !grid.sources[key].selected; // Toggle selection
				grid.sources[key].logSource(); // Optional: log position
			}
		}
		else if (grid.nodes[key].isClicked(mouseX, mouseY)) {
                grid.nodes[key].selected = !grid.nodes[key].selected; // Toggle selection
                grid.nodes[key].logNode(); // Optional: log position
            }
	}
}

window.setup = setup;
window.draw = draw;
// window.mousePressed = mousePressed