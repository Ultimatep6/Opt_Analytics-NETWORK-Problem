function Connection(nodeA, nodeB, grid) {
	this.a = nodeA;
	this.b = nodeB;

	this.flows = {};
	this.capacity = 100; // Default capacity

	this.grid = grid;

	this.addFlow = function(matFlow, amount) {
		if (this.flows[matFlow]) {
			this.flows[matFlow] += amount;
		} else {
			this.flows[matFlow] = amount;
		}
		if (!(matFlow in this.grid.flowColors)) {
			this.grid.addMaterialColor(matFlow, color(random(255), random(255), random(255)));
		}
	}
}

