function Node(i, j) {
	this.i = i;
	this.j = j;

	this.x = this.i * (gridWidth / cols) + (gridWidth / cols) / 2;
	this.y = this.j * (gridHeight / rows) + (gridHeight / rows) / 2;

	this.radius = 20;
	this.selected = false;

	this.show = function() {
		fill(this.selected ? 'red' : 0);
		stroke(255);
		ellipse(this.x, this.y, this.radius, this.radius);
	};

	this.hide = function() {
		fill(255);
		stroke(255);
		ellipse(this.x, this.y, this.radius, this.radius);
	}

	this.isClicked = function(mx, my) {
		let x = this.x;
		let y = this.y;
		return dist(mx, my, x, y) < this.radius / 2;
	};

	this.logNode = function() {
		console.log("Node at (" + this.i + ", " + this.j + ")");
	}
}